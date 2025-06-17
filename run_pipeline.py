import numpy as np
from unyt import angstrom, kpc, Mpc, Msun, Gyr, km, s, yr, arcsecond
import h5py
from astropy.cosmology import Planck15 as cosmo
import time
import argparse
import os

from synthesizer import check_openmp
print('OpenMP enabled:', check_openmp() )

from mpi4py import MPI as mpi
import multiprocessing as mp

from synthesizer.emission_models import IntrinsicEmission
from synthesizer.grid import Grid
from synthesizer.instruments import UVJ, FilterCollection, Instrument
from synthesizer.pipeline import Pipeline
from synthesizer.load_data.load_flares import load_FLARES
from synthesizer.particle import Galaxy
from synthesizer.particle import Stars, Gas, BlackHoles
from synthesizer.kernel_functions import Kernel

from my_emission_models import FLARESLOSCombinedEmission

from utils import SPECTRA_KEYS

def _print(*args, **kwargs):
    """Overload print with rank info."""
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"[{str(rank).zfill(len(str(size)) + 1)}]: ", end="")
    print(*args, **kwargs)
    
    
def _get_galaxy(gal_ind, master_file_path, reg, snap, z):
    """
    Get a galaxy from the master file.

    Args:
        gal_ind (int): The index of the galaxy to get.
        master_file_path (str): The path to the master file.
        reg (str): The region to use.
        snap (str): The snapshot to use.
        z (float): The redshift of the snapshot.
    """
    # Get the galaxy data we need from the master file
    with h5py.File(master_file_path, "r") as hdf:
        reg_grp = hdf[reg]
        snap_grp = reg_grp[snap]
        gal_grp = snap_grp["Galaxy"]
        part_grp = snap_grp["Particle"]

        # Get the group and subgrp ids
        group_id = gal_grp["GroupNumber"][gal_ind]
        subgrp_id = gal_grp["SubGroupNumber"][gal_ind]

        # Get this galaxy's beginning and ending indices for stars
        s_len = gal_grp["S_Length"][...]
        start = np.sum(s_len[:gal_ind])
        end = np.sum(s_len[: gal_ind + 1])

        # Get this galaxy's beginning and ending indices for gas
        g_len = gal_grp["G_Length"][...]
        start_gas = np.sum(g_len[:gal_ind])
        end_gas = np.sum(g_len[: gal_ind + 1])

        # Get this galaxy's beginning and ending indices for black holes
        bh_len = gal_grp["BH_Length"][...]
        start_bh = np.sum(bh_len[:gal_ind])
        end_bh = np.sum(bh_len[: gal_ind + 1])

        # Get the star data
        star_pos = part_grp["S_Coordinates"][:, start:end].T / (1 + z) * Mpc
        star_mass = part_grp["S_Mass"][start:end] * Msun * 10**10
        star_init_mass = part_grp["S_MassInitial"][start:end] * Msun * 10**10
        star_age = part_grp["S_Age"][start:end] * Gyr
        star_met = part_grp["S_Z_smooth"][start:end]
        star_sml = part_grp["S_sml"][start:end] * Mpc
        star_vel = part_grp["S_Vel"][:, start:end].T * km / s

        # Get the gas data
        gas_pos = (
            part_grp["G_Coordinates"][:, start_gas:end_gas].T / (1 + z) * Mpc
        )
        gas_mass = part_grp["G_Mass"][start_gas:end_gas] * Msun * 10**10
        gas_met = part_grp["G_Z_smooth"][start_gas:end_gas]
        gas_sml = part_grp["G_sml"][start_gas:end_gas] * Mpc

        # Get the black hole data
        bh_pos = (
            part_grp["BH_Coordinates"][:, start_bh:end_bh].T / (1 + z) * Mpc
        )
        bh_mass = part_grp["BH_Mass"][start_bh:end_bh] * Msun * 10**10
        bh_mdot = (
            part_grp["BH_Mdot"][start_bh:end_bh]
            * (
                6.445909132449984 * 10**23
            )  # Unit conversion issue, need this
            * Msun
            / yr
        )

        # Get the centre of potential
        centre = gal_grp["COP"][:].T[gal_ind, :] / (1 + z) * Mpc

        # Compute the angular radii of each star in arcseconds
        radii = (np.linalg.norm(star_pos - centre, axis=1)).to("kpc")
        star_ang_rad = (
            radii.value * cosmo.arcsec_per_kpc_proper(z).value * arcsecond
        )

    # Define a mask to get a 30 kpc aperture
    mask = radii < 30 * kpc

    # Early exist if there are fewer than 100 stars
    if np.sum(mask) < 100:
        return None

    gal = Galaxy(
        name=f"{reg}_{snap}_{gal_ind}_{group_id}_{subgrp_id}",
        redshift=z,
        stars=Stars(
            initial_masses=star_init_mass[mask],
            current_masses=star_mass[mask],
            ages=star_age[mask],
            metallicities=star_met[mask],
            redshift=z,
            coordinates=star_pos[mask, :],
            velocities=star_vel[mask, :],
            smoothing_lengths=star_sml[mask],
            centre=centre,
            young_tau_v=star_met[mask] / 0.01,
            angular_radii=star_ang_rad[mask],
            radii=radii[mask].value,
        ),
        gas=Gas(
            masses=gas_mass,
            metallicities=gas_met,
            redshift=z,
            coordinates=gas_pos,
            smoothing_lengths=gas_sml,
            centre=centre,
        ),
        black_holes=BlackHoles(
            masses=bh_mass,
            accretion_rates=bh_mdot,
            coordinates=bh_pos,
            redshift=z,
            centre=centre,
        ),
    )

    # Calculate the DTM, we'll need it later
    gal.calculate_dust_to_metal_vijayan19()

    # Compute what we can compute out the gate and attach it to the galaxy
    # for later use
    gal.gas.half_mass_radius = gal.gas.get_half_mass_radius()
    gal.gas.mass_radii = {
        0.2: gal.gas.get_attr_radius("masses", frac=0.2),
        0.8: gal.gas.get_attr_radius("masses", frac=0.8),
    }
    gal.gas.half_dust_radius = gal.gas.get_attr_radius("dust_masses")
    gal.stars.half_mass_radius = gal.stars.get_half_mass_radius()
    gal.stars.mass_radii = {
        0.2: gal.stars.get_attr_radius("current_masses", frac=0.2),
        0.8: gal.stars.get_attr_radius("current_masses", frac=0.8),
    }

    # Calculate the 3D and 1D velocity dispersions
    gal.stars.vel_disp1d = np.array(
        [
            np.std(gal.stars.velocities[:, 0], ddof=0),
            np.std(gal.stars.velocities[:, 1], ddof=0),
            np.std(gal.stars.velocities[:, 2], ddof=0),
        ]
    )
    gal.stars.vel_disp3d = np.std(
        np.sqrt(np.sum(gal.stars.velocities**2, axis=1)), ddof=0
    )

    return gal


def get_flares_galaxies(
    master_file_path,
    region,
    snap,
    nthreads,
    comm,
    rank,
    size,
):
    """
    Get Galaxy objects for FLARES galaxies.

    Args:
        master_file_path (str): The path to the master file.
        region (int): The region to use.
        snap (str): The snapshot to use.
        filter_collection (FilterCollection): The filter collection to use.
        emission_model (StellarEmissionModel): The emission model to use.
    """
    
    # Get the region tag
    reg = str(region).zfill(2)

    # Get redshift from the snapshot tag
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))
    #if snap == "008_z007p000":
    #    z = 7.29
    #    _print("Shifted redshift to 7.29 for snapshot 008_z007p000")

    # How many galaxies are there?
    with h5py.File(master_file_path, "r") as hdf:
        reg_grp = hdf[reg]
        snap_grp = reg_grp[snap]
        gal_grp = snap_grp["Galaxy"]
        s_lens = gal_grp["S_Length"][:]
        n_gals = len(s_lens)

    # Early exist if there are no galaxies
    if n_gals == 0:
        return []

    # Randomize the order of galaxies
    np.random.seed(42)
    order = np.random.permutation(n_gals)

    # Distribute galaxies by number of particles
    parts_per_rank = np.zeros(size, dtype=int)
    gals_on_rank = {rank: [] for rank in range(size)}
    for i in order:
        if s_lens[i] < 100:
            continue
        select = np.argmin(parts_per_rank)
        gals_on_rank[select].append(i)
        parts_per_rank[select] += s_lens[i]

    # Prepare the arguments for each galaxy on this rank
    args = [
        (gal_ind, master_file_path, reg, snap, z)
        for gal_ind in gals_on_rank[rank]
    ]

    # Get all the galaxies using multiprocessing
    with mp.Pool(nthreads) as pool:
        galaxies = pool.starmap(_get_galaxy, args)

    # Remove any Nones
    galaxies = [gal for gal in galaxies if gal is not None]

    return galaxies


def get_webb_filters(filepath):
    """Get the filter collection."""
    
    # Check if the filter collection file already exists
    if os.path.exists(filepath):
        filters = FilterCollection(path=filepath)
    else:

        lam = np.linspace(10**3, 10**5, 1000) * angstrom
        filters = FilterCollection(
            filter_codes=[
                f"JWST/NIRCam.{f}"
                for f in ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
            ],
            new_lam=lam,
        )

        # Write the filter collection
        filters.write_filters(path=filepath)

    return filters


def get_uvj_filters(filepath):
    """Get the filter collection."""
    
    # Check if the filter collection file already exists
    if os.path.exists(filepath):
        filters = FilterCollection(path=filepath)
    else:

        lam = np.linspace(10**3, 10**5, 1000) * angstrom
        filters = UVJ(new_lam=lam)

        # Write the filter collection
        filters.write_filters(path=filepath)

    return filters

def get_UV_slopes(obj):
    """
    Get the UV slope of the galaxy.

    Args:
        obj (Galaxy/Stars/BlackHoles): The object to get the UV slope for.

    Returns:
        float: The UV slope.
    """
    # Dictionary to hold the slopes
    slopes = {}

    # Loop over the spectra
    for spec_type, d in obj.spectra.items():
        slopes[spec_type] = obj.spectra[spec_type].measure_beta(
            window=(1500 * angstrom, 3000 * angstrom)
        )

    return slopes


def get_IR_slopes(obj):
    """
    Get the IR slopes of the galaxy.

    Args:
        obj (Galaxy/Stars/BlackHoles): The object to get the IR slopes for.

    Returns:
        float: The IR slopes.
    """
    # Dictionary to hold the slopes
    slopes = {}

    # Loop over the spectra
    for spec_type, d in obj.spectra.items():
        slopes[spec_type] = obj.spectra[spec_type].measure_beta(
            window=(4400 * angstrom, 7500 * angstrom)
        )

    return slopes

def get_emission_model(
    grid_name,
    grid_dir,
    fesc=0.0,
    fesc_ly_alpha=1.0,
    agn_template_file="vandenberk_agn_template.txt",
    save_spectra=SPECTRA_KEYS,
):
    """Get the emission model to use for the observations."""
    grid = Grid(
        grid_name,
        grid_dir,
        lam_lims=(900 * angstrom, 6 * 10**5 * angstrom),
    )
    model = FLARESLOSCombinedEmission(
        grid,
        agn_template_file,
    )

    # Limit the spectra to be saved
    model.save_spectra(*save_spectra)

    return model


def get_aperture_phot(gal, app_rs=[1, 3, 5, 10, 20, 30, 40, 50, 70, 100]):
    # Define dict to hold results
    result_dict = {}

    # Loop over all particle_photo_* on your stars.
    for key, photcol in gal.stars.photo_fluxes.items():
        print(key, photocol)
        if key == "total":
            
            # Loop over appertures 
            for app_r in app_rs:

                mask = gal.stars.radii < app_r * 1e-12 * kpc
                result_dict[f'{app_r} pkpc'] = photcol[mask]

    return result_dict


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Derive synthetic observations for FLARES."
    )

    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use.",
    )

    # Parse the arguments
    args = parser.parse_args()
    nthreads = args.nthreads
    
    # Get the grid
    grid_dir = '/cosma7/data/dp276/dc-newm1/synthesizer_data/grids/'
    grid_name = 'bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c23.01-sps'
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Emission model
    model = get_emission_model(grid_name, grid_dir, fesc=0.1)
    
    # We can use run the Pipeline for multiple grids by setting:
    #model.set_grid(new_grid, set_all=True)

    # Get the filters
    # Note that if running on a HPC you will need to make these filter files 
    # seperately to this code
    filters = get_webb_filters("/cosma7/data/dp276/dc-newm1/synthesizer_data/rubies_filters.hdf5")

    # Instatiate the instruments
    inst = Instrument("RUBIES", filters=filters, resolution=0.1 * kpc)
    print(inst)

    # Get the galaxies
    master_file_path = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"
    region = "04" # region
    snap =  "008_z007p000" # the fourth snapshot

    # Print n CPUs for reference
    cpuCount = os.cpu_count()
    print("Number of CPUs in the system:", cpuCount)

    # Get MPI info
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
        
    # Get the galaxies
    read_start = time.time()
    galaxies = get_flares_galaxies(
        master_file_path,
        region,
        snap,
        nthreads,
        comm,
        rank,
        size,
    )
    read_end = time.time()
    _print(
        f"Creating {len(galaxies)} galaxies took "
        f"{read_end - read_start:.2f} seconds."
    )

    # Start Pipeline object
    pipeline = Pipeline(
        emission_model=model,
        #instruments=instruments,
        nthreads=nthreads,
        verbose=1,
        comm=mpi.COMM_WORLD,
    )

    pipeline.add_galaxies(galaxies)
    pipeline.get_spectra()
    pipeline.get_observed_spectra(cosmo=cosmo)
    pipeline.get_lines(line_ids=grid.available_lines)
    pipeline.get_observed_lines(cosmo)

    # Photometry
    pipeline.get_photometry_luminosities(inst)
    pipeline.get_photometry_fluxes(inst)

    # Get the SPH kernel
    #sph_kernel = Kernel()
    #kernel = sph_kernel.get_kernel()

    # If imaging is needed
    #pipeline.get_images_luminosity(inst, fov=50 * kpc, kernel=kernel)
    #pipeline.get_images_flux(inst, fov=50 * kpc, kernel=kernel)
    
    # Add galaxy info
    pipeline.add_analysis_func(lambda gal: gal.region, "Region")
    pipeline.add_analysis_func(lambda gal: gal.grp_id, "GroupID")
    pipeline.add_analysis_func(lambda gal: gal.subgrp_id, "SubGroupID")
    pipeline.add_analysis_func(lambda gal: gal.weight, "RegionWeight")
    pipeline.add_analysis_func(lambda gal: gal.master_index, "MasterRegionIndex")
    pipeline.add_analysis_func(lambda gal: gal.redshift, "Redshift")
    
    # Attach apertures from images
        #for app in ["0p2", "0p4"]:
        #    for spec in SPECTRA_KEYS:
        #        for filt in FILTER_CODES:
        #            apps[app][spec][filt].append(
        #                gal.images_fnu[spec].app_fluxes[filt][app]
        #            )
        
    
    # Add photometry with apertures
    pipeline.add_analysis_func(
        lambda gal: get_aperture_phot(gal),
        "ApertureTotalPhotometry"
    )
    
    # Get slopes
    pipeline.add_analysis_func(
         lambda gal: get_UV_slopes(gal.stars),
         "Stars/UVSlope",
         )
    pipeline.add_analysis_func(
         lambda gal: get_IR_slopes(gal.stars),
         "Stars/IRSlope",
    )

    pipeline.run()
    pipeline.write(f"./results/RUBIES_COMP_{str(region).zfill(2)}_{snap}.hdf5", verbose=0)




