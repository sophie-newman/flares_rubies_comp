"""A script to derive synthetic observations for FLARES."""

import argparse
import pickle
import time
import os
import multiprocessing as mp
import numpy as np
import h5py
from unyt import Gyr, yr, Mpc, Msun, arcsecond, angstrom, kpc, km, s
from astropy.cosmology import Planck15 as cosmo
from mpi4py import MPI as mpi
from utils import (
    write_dataset_recursive,
    _print,
    sort_data_recursive,
    SPECTRA_KEYS,
)
from utils import RUBIES_FILTER_CODES as FILTER_CODES
import webbpsf

from synthesizer.particle import Stars, Gas, BlackHoles
from synthesizer.particle import Galaxy
from synthesizer.filters import FilterCollection
from synthesizer import Grid
from synthesizer.kernel_functions import Kernel
from synthesizer._version import __version__
from synthesizer.conversions import angular_to_spatial_at_z

from combined_emission_model import FLARESLOSCombinedEmission

# Silence warnings (only because we now what we're doing)
import warnings

warnings.filterwarnings("ignore")

# Msun needs to be respected
Msun = Msun.in_base("galactic")


def recursive_gather(data, comm, root=0):
    """
    Recursively collect data from all ranks onto the root.

    This function is effectively just comm.gather but will ensure the
    gather is performed at the root of any passed dictionaries to avoid
    overflows and any later recursive concatenation issues.

    Args:
        data (any): The data to gather.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.
    """
    # If we don't have a dict, just gather the data straight away
    if not isinstance(data, dict):
        collected_data = comm.gather(data, root=root)
        if comm.rank == root:
            collected_data = [l for l in collected_data if len(l) > 0]
            if len(collected_data) > 0:
                return np.concatenate(collected_data)
            else:
                return []
        else:
            return []

    # Recurse through the whole dict communicating an lists or
    # arrays we hit along the way
    def _gather(d, comm, root):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, (list, np.ndarray)):
                collected_data = comm.gather(v, root=root)
                if comm.rank == root:
                    collected_data = [l for l in collected_data if len(l) > 0]
                    if len(collected_data) > 0:
                        new_d[k] = np.concatenate(collected_data)
                    else:
                        new_d[k] = []
                else:
                    new_d[k] = []
            elif isinstance(v, dict):
                new_d[k] = _gather(v, comm, root)
            else:
                raise ValueError(f"Unexpected type {type(v)}")
        return new_d

    return _gather(data, comm, root)


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
    gal.dust_to_metal_vijayan19()

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
    if snap == "008_z007p000":
        z = 7.29
        _print("Shifted redshift to 7.29 for snapshot 008_z007p000")

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


def get_flares_filters(filepath):
    """Get the filter collection."""
    # Check if the filter collection file already exists
    if os.path.exists(filepath):
        filters = FilterCollection(path=filepath)
    else:
        tophats = {
            "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
        }

        # Create the FilterCollection
        filters = FilterCollection(
            filter_codes=FILTER_CODES[1:],
            tophat_dict=tophats,
        )

        # Write the filter collection
        filters.write_filters(path=filepath)

    return filters


def get_grid(grid_name, grid_dir, filters):
    """Get a Synthesizer Grid."""
    grid = Grid(
        grid_name,
        grid_dir,
        lam_lims=(1000 * angstrom, 10**6 * angstrom),
    )

    filters.resample_filters(new_lam=grid.lam)

    return grid, filters


def get_emission_model(
    grid,
    fesc=0.0,
    fesc_ly_alpha=1.0,
    agn_template_file="vandenberk_agn_template.txt",
    save_spectra=(
        "reprocessed",
        "attenuated",
        "young_reprocessed",
        "old_reprocessed",
        "young_attenuated",
        "old_attenuated",
        "agn_intrinsic",
        "agn_attenuated",
        "combined_intrinsic",
        "total",
    ),
):
    """Get a StellarEmissionModel."""
    model = FLARESLOSCombinedEmission(
        agn_template_file,
        grid,
        fesc=fesc,
        fesc_ly_alpha=fesc_ly_alpha,
    )

    # Limit the spectra to be saved
    model.save_spectra(*save_spectra)

    return model


def get_kernel():
    """Get a Kernel."""
    return Kernel()


def get_psfs(filter_codes, filepath):
    """Get the PSFs for each filter."""
    # Check if we can load a file
    if os.path.exists(filepath):
        psfs = {}
        with h5py.File(filepath, "r") as hdf:
            for filt in filter_codes:
                # Skip an without JWST
                if "JWST" not in filt:
                    continue
                psf = hdf[filt][...]
                psfs[filt] = psf
    else:
        # Ok we need to make them
        psfs = {}
        for filt in filter_codes:
            # Skip an without JWST
            if "JWST" not in filt:
                continue
            if "NIRCam" in filt:
                nc = webbpsf.NIRCam()
                nc.filter = filt.split(".")[-1]
                psf = nc.calc_psf(oversample=2)
                psfs[filt] = psf[0].data
            elif "MIRI" in filt:
                miri = webbpsf.MIRI()
                miri.filter = filt.split(".")[-1]
                psf = miri.calc_psf(oversample=2)
                psfs[filt] = psf[0].data

        # Write out the PSFs for later use
        with h5py.File(filepath, "w") as hdf:
            for filt, psf in psfs.items():
                hdf.create_dataset(filt, data=psf)

    # Create a fake PSF of ones for the UV1500 filter
    psfs["UV1500"] = np.ones((101, 101))

    return psfs


def get_images(
    gal,
    spec_key,
    agn_spec_key,
    kernel,
    nthreads,
    psfs,
    cosmo,
):
    """Get an image of the galaxy in an array of filters."""
    # Setup the image properties
    ang_res = 0.031 * arcsecond
    kpc_res = (
        ang_res.value / cosmo.arcsec_per_kpc_proper(gal.redshift).value
    ) * kpc
    fov = 30 * kpc

    # Get the image
    imgs = gal.get_images_flux(
        resolution=kpc_res,
        fov=fov,
        img_type="smoothed",
        stellar_photometry=spec_key,
        blackhole_photometry=agn_spec_key,
        kernel=kernel,
        nthreads=nthreads,
    )

    # Apply the PSFs with a super and then down sample to remove any convolution
    # issues due to the resolution.
    imgs.supersample(2)
    psf_imgs = imgs.apply_psfs(psfs)
    psf_imgs.downsample(0.5)

    # Apply the 0.2" and 0.4" apertures
    ang_apertures = np.array([0.2, 0.32, 0.4, 0.5]) * arcsecond
    kpc_apertures = angular_to_spatial_at_z(ang_apertures, cosmo, gal.redshift)
    app_flux = {}
    for filt in FILTER_CODES:
        app_flux.setdefault(filt, {})
        for ap, lab in zip(kpc_apertures, ["0p2", "0p32", "0p4", "0p5"]):
            app_flux[filt][lab] = float(
                psf_imgs[filt]
                .get_signal_in_aperture(ap.to(Mpc), nthreads=nthreads)
                .value
            )

    # Attach apertures to image
    psf_imgs.app_fluxes = app_flux

    # Compute and store the fluxes based on the images
    img_fluxes = {}
    for filt in FILTER_CODES:
        img_fluxes[filt] = np.sum(psf_imgs[filt].arr)

    # Attach the fluxes to the images
    psf_imgs.fluxes = img_fluxes

    return psf_imgs


def analyse_galaxy(
    gal,
    emission_model,
    grid,
    kern,
    nthreads,
    filters,
    cosmo,
    psfs,
):
    """
    Analyse a galaxy.

    This will generate all spectra and photometry for a galaxy.

    Args:
        gal (Galaxy): The galaxy to analyse.
        emission_model (StellarEmissionModel): The emission model to use.
        kern (Kernel): The kernel to use.
        filters (FilterCollection): The filter collection to use.
        cosmo (astropy.cosmology): The cosmology to use.
    """
    # Get the los tau_v
    if gal.stars.tau_v is None:
        gal.get_stellar_los_tau_v(
            kappa=0.0795,
            kernel=kern.get_kernel(),
            force_loop=False,
            nthreads=nthreads,
        )
    if gal.black_holes.tau_v is None:
        gal.get_black_hole_los_tau_v(
            kappa=0.0795,
            kernel=kern.get_kernel(),
            force_loop=False,
            nthreads=nthreads,
        )

    # Get the SFZH
    if gal.stars.sfzh is None:
        gal.stars.get_sfzh(grid, nthreads=nthreads)

    # Get the spectra
    gal.get_spectra(
        emission_model,
        nthreads=nthreads,
    )

    # Get the observed spectra
    gal.get_observed_spectra(cosmo)

    # Get the photometry
    gal.get_photo_luminosities(filters, verbose=False)
    gal.get_photo_fluxes(filters, verbose=False)

    # Compute the half-light radius on each filter
    gal.stars.half_light_radii = {}
    for spec in gal.stars.particle_photo_fnu.keys():
        gal.stars.half_light_radii[spec] = {}
        for filt in filters.filter_codes:
            # Get the half light radius
            gal.stars.half_light_radii[spec][
                filt
            ] = gal.stars.get_half_flux_radius(spec, filt)

    # Get the 95% light radius
    gal.stars.light_radii_95 = {}
    for spec in gal.stars.particle_photo_fnu.keys():
        gal.stars.light_radii_95[spec] = {}
        for filt in filters.filter_codes:
            # Get the light radius
            gal.stars.light_radii_95[spec][filt] = gal.stars.get_flux_radius(
                spec, filt, frac=0.95
            )

    # Get the 80% light radius
    gal.stars.light_radii_80 = {}
    for spec in gal.stars.particle_photo_fnu.keys():
        gal.stars.light_radii_80[spec] = {}
        for filt in filters.filter_codes:
            # Get the light radius
            gal.stars.light_radii_80[spec][filt] = gal.stars.get_flux_radius(
                spec, filt, frac=0.8
            )

    # Get the 20% light radius
    gal.stars.light_radii_20 = {}
    for spec in gal.stars.particle_photo_fnu.keys():
        gal.stars.light_radii_20[spec] = {}
        for filt in filters.filter_codes:
            # Get the light radius
            gal.stars.light_radii_20[spec][filt] = gal.stars.get_flux_radius(
                spec, filt, frac=0.2
            )

    # Get the images
    gal.flux_imgs = {}
    imgs = get_images(
        gal,
        "reprocessed",
        "agn_intrinsic",
        kernel=kern.get_kernel(),
        nthreads=nthreads,
        psfs=psfs,
        cosmo=cosmo,
    )
    gal.flux_imgs["reprocessed"] = imgs
    imgs = get_images(
        gal,
        None,
        "agn_intrinsic",
        kernel=kern.get_kernel(),
        nthreads=nthreads,
        psfs=psfs,
        cosmo=cosmo,
    )
    gal.flux_imgs["agn_reprocessed"] = imgs
    imgs = get_images(
        gal,
        "reprocessed",
        None,
        kernel=kern.get_kernel(),
        nthreads=nthreads,
        psfs=psfs,
        cosmo=cosmo,
    )
    gal.flux_imgs["stellar_reprocessed"] = imgs
    imgs = get_images(
        gal,
        "attenuated",
        "agn_attenuated",
        kernel=kern.get_kernel(),
        nthreads=nthreads,
        psfs=psfs,
        cosmo=cosmo,
    )
    gal.flux_imgs["attenuated"] = imgs
    imgs = get_images(
        gal,
        None,
        "agn_attenuated",
        kernel=kern.get_kernel(),
        nthreads=nthreads,
        psfs=psfs,
        cosmo=cosmo,
    )
    gal.flux_imgs["agn_attenuated"] = imgs
    imgs = get_images(
        gal,
        "attenuated",
        None,
        kernel=kern.get_kernel(),
        nthreads=nthreads,
        psfs=psfs,
        cosmo=cosmo,
    )
    gal.flux_imgs["stellar_attenuated"] = imgs

    return gal


def write_results(galaxies, path, grid_name, filters, comm, rank, size):
    """Write the results to a file."""
    # Get the redshift from the first galaxy
    if len(galaxies) > 0:
        z = galaxies[0].redshift
    else:
        z = None

    # Setup the structure of all output dicts and lists
    fnus = {}
    fluxes = {}
    rf_fluxes = {}
    imgs = {}
    img_fluxes = {}
    uv_slopes = {}
    ir_slopes = {}
    sizes = {}
    sizes_95 = {}
    sizes_80 = {}
    sizes_20 = {}
    apps = {}
    apps["0p2"] = {}
    apps["0p32"] = {}
    apps["0p4"] = {}
    apps["0p5"] = {}
    gal_ids = []
    gas_sizes = []
    gas_sizes_80 = []
    gas_sizes_20 = []
    dust_sizes = []
    group_ids = []
    subgroup_ids = []
    indices = []
    sfzhs = []
    vel_disp_1d = []
    vel_disp_3d = []
    for spec in [
        "reprocessed",
        "attenuated",
        "agn_reprocessed",
        "stellar_reprocessed",
        "agn_attenuated",
        "stellar_attenuated",
        "young_reprocessed",
        "old_reprocessed",
        "young_attenuated",
        "old_attenuated",
        "agn_intrinsic",
        "combined_intrinsic",
        "total",
    ]:
        fnus[spec] = []
        fluxes[spec] = {}
        rf_fluxes[spec] = {}
        imgs[spec] = {}
        img_fluxes[spec] = {}
        uv_slopes[spec] = []
        ir_slopes[spec] = []
        sizes[spec] = {}
        sizes_95[spec] = {}
        sizes_80[spec] = {}
        sizes_20[spec] = {}
        apps["0p2"][spec] = {}
        apps["0p32"][spec] = {}
        apps["0p4"][spec] = {}
        apps["0p5"][spec] = {}
        for filt in FILTER_CODES + ["UV1500"]:
            imgs[spec][filt] = []
            img_fluxes[spec][filt] = []
            fluxes[spec][filt] = []
            rf_fluxes[spec][filt] = []
            sizes[spec][filt] = []
            sizes_95[spec][filt] = []
            sizes_80[spec][filt] = []
            sizes_20[spec][filt] = []
            apps["0p2"][spec][filt] = []
            apps["0p32"][spec][filt] = []
            apps["0p4"][spec][filt] = []
            apps["0p5"][spec][filt] = []

    # Loop over galaxies and unpacking all the data we'll write out
    for gal in galaxies:
        # Get the group and subgroup ids
        indices.append(int(gal.name.split("_")[3]))
        group_ids.append(int(gal.name.split("_")[4]))
        subgroup_ids.append(int(gal.name.split("_")[5]))
        gal_ids.append(gal.name)

        # Unpack the gas size information
        gas_sizes.append(gal.gas.half_mass_radius)
        gas_sizes_80.append(gal.gas.mass_radii[0.8])
        gas_sizes_20.append(gal.gas.mass_radii[0.2])
        dust_sizes.append(gal.gas.half_dust_radius)

        # Unpack the velocity dispersions
        vel_disp_1d.append(gal.stars.vel_disp1d)
        vel_disp_3d.append(gal.stars.vel_disp3d)

        # Get the SFZH arrays
        sfzhs.append(gal.stars.sfzh)

        # Get the integrated observed spectra
        for key, spec in gal.stars.spectra.items():
            fnus[key].append(spec._fnu)
        for key, spec in gal.black_holes.spectra.items():
            fnus[key].append(spec._fnu)
        for key, spec in gal.spectra.items():
            fnus[key].append(spec._fnu)

        # Get the images
        for spec in [
            "reprocessed",
            "attenuated",
            "agn_reprocessed",
            "stellar_reprocessed",
            "agn_attenuated",
            "stellar_attenuated",
        ]:
            for key in FILTER_CODES:
                imgs[spec][key].append(gal.flux_imgs[spec][key].arr)

        # Get the photometry
        for key, photcol in gal.stars.photo_fluxes.items():
            for filt, phot in photcol.items():
                fluxes[key][filt].append(phot)
        for key, photcol in gal.black_holes.photo_fluxes.items():
            for filt, phot in photcol.items():
                fluxes[key][filt].append(phot)
        for key, photcol in gal.photo_fluxes.items():
            for filt, phot in photcol.items():
                fluxes[key][filt].append(phot)

        # Get the rest frame photometry
        for key, photcol in gal.stars.photo_luminosities.items():
            for filt, phot in photcol.items():
                rf_fluxes[key][filt].append(phot)
        for key, photcol in gal.black_holes.photo_luminosities.items():
            for filt, phot in photcol.items():
                rf_fluxes[key][filt].append(phot)
        for key, photcol in gal.photo_luminosities.items():
            for filt, phot in photcol.items():
                rf_fluxes[key][filt].append(phot)

        # Get slopes
        for key, spectra in gal.stars.spectra.items():
            uv_slopes[key].append(
                spectra.measure_beta(window=(1500 * angstrom, 3000 * angstrom))
            )
            ir_slopes[key].append(
                spectra.measure_beta(window=(4400 * angstrom, 7500 * angstrom))
            )

        # Get the sizes
        for spec in gal.stars.half_light_radii.keys():
            for filt in gal.stars.half_light_radii[spec].keys():
                sizes[spec][filt].append(
                    gal.stars.half_light_radii[spec][filt]
                )
        for spec in gal.stars.light_radii_95.keys():
            for filt in gal.stars.light_radii_95[spec].keys():
                sizes_95[spec][filt].append(
                    gal.stars.light_radii_95[spec][filt]
                )
        for spec in gal.stars.light_radii_80.keys():
            for filt in gal.stars.light_radii_80[spec].keys():
                sizes_80[spec][filt].append(
                    gal.stars.light_radii_80[spec][filt]
                )
        for spec in gal.stars.light_radii_20.keys():
            for filt in gal.stars.light_radii_20[spec].keys():
                sizes_20[spec][filt].append(
                    gal.stars.light_radii_20[spec][filt]
                )

        # Attach apertures from images
        for app in ["0p2", "0p32", "0p4", "0p5"]:
            for spec in [
                "reprocessed",
                "attenuated",
                "agn_reprocessed",
                "stellar_reprocessed",
                "agn_attenuated",
                "stellar_attenuated",
            ]:
                for filt in FILTER_CODES:
                    apps[app][spec][filt].append(
                        gal.flux_imgs[spec].app_fluxes[filt][app]
                    )

        # Get the fluxes from the images
        for spec in [
            "reprocessed",
            "attenuated",
            "agn_reprocessed",
            "stellar_reprocessed",
            "agn_attenuated",
            "stellar_attenuated",
        ]:
            for filt in FILTER_CODES:
                img_fluxes[spec][filt].append(gal.flux_imgs[spec].fluxes[filt])

    # Collect output data onto rank 0, this is done recursively with the results
    # of the gather being concatenated at the root
    fnus = recursive_gather(fnus, comm, root=0)
    fluxes = recursive_gather(fluxes, comm, root=0)
    rf_fluxes = recursive_gather(rf_fluxes, comm, root=0)
    group_ids = recursive_gather(group_ids, comm, root=0)
    subgroup_ids = recursive_gather(subgroup_ids, comm, root=0)
    gal_ids = recursive_gather(gal_ids, comm, root=0)
    indices = recursive_gather(indices, comm, root=0)
    uv_slopes = recursive_gather(uv_slopes, comm, root=0)
    ir_slopes = recursive_gather(ir_slopes, comm, root=0)
    sizes = recursive_gather(sizes, comm, root=0)
    sizes_95 = recursive_gather(sizes_95, comm, root=0)
    sizes_80 = recursive_gather(sizes_80, comm, root=0)
    sizes_20 = recursive_gather(sizes_20, comm, root=0)
    gas_sizes = recursive_gather(gas_sizes, comm, root=0)
    gas_sizes_80 = recursive_gather(gas_sizes_80, comm, root=0)
    gas_sizes_20 = recursive_gather(gas_sizes_20, comm, root=0)
    dust_sizes = recursive_gather(dust_sizes, comm, root=0)
    sfzhs = recursive_gather(sfzhs, comm, root=0)
    apps = recursive_gather(apps, comm, root=0)
    img_fluxes = recursive_gather(img_fluxes, comm, root=0)
    vel_disp_1d = recursive_gather(vel_disp_1d, comm, root=0)
    vel_disp_3d = recursive_gather(vel_disp_3d, comm, root=0)
    imgs = recursive_gather(imgs, comm, root=0)

    # Early exit if we're not rank 0
    if rank != 0:
        return

    # Remove any empty keys
    def _remove_empty(d):
        if isinstance(d, dict):
            return {k: _remove_empty(v) for k, v in d.items() if len(v) > 0}
        elif isinstance(d, list):
            return [v for v in d if len(v) > 0]
        return d

    fnus = _remove_empty(fnus)
    fluxes = _remove_empty(fluxes)
    rf_fluxes = _remove_empty(rf_fluxes)
    uv_slopes = _remove_empty(uv_slopes)
    ir_slopes = _remove_empty(ir_slopes)
    sizes = _remove_empty(sizes)
    sizes_95 = _remove_empty(sizes_95)
    sizes_80 = _remove_empty(sizes_80)
    sizes_20 = _remove_empty(sizes_20)
    apps = _remove_empty(apps)
    img_fluxes = _remove_empty(img_fluxes)
    imgs = _remove_empty(imgs)

    # Get the units for each dataset
    units = {
        "fnu": "nJy",
        "flux": "erg/s/cm^2/Hz",
        "hlr": "kpc",
        "sfzh": "Msun",
        "vel": "km/s",
    }

    # Get the indices that will sort the data to match the master file
    sort_indices = np.argsort(indices)

    # Write output out to file
    with h5py.File(path, "w") as hdf:
        # Store the grid used and the Synthesizer version
        hdf.attrs["Grid"] = grid_name
        hdf.attrs["SynthesizerVersion"] = __version__
        hdf.attrs["Redshift"] = z

        # Write the group and subgroup ids
        write_dataset_recursive(
            hdf,
            sort_data_recursive(group_ids, sort_indices),
            "GroupNumber",
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(subgroup_ids, sort_indices),
            "SubGroupNumber",
        )
        write_dataset_recursive(
            hdf,
            [
                s.encode("utf-8")
                for s in sort_data_recursive(gal_ids, sort_indices)
            ],
            "GalaxyID",
        )

        # Write the velocity dispersions
        write_dataset_recursive(
            hdf,
            sort_data_recursive(vel_disp_1d, sort_indices),
            "1DVelocityDispersion",
            units=units["vel"],
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(vel_disp_3d, sort_indices),
            "3DVelocityDispersion",
            units=units["vel"],
        )

        # Write the integrated observed spectra
        write_dataset_recursive(
            hdf,
            sort_data_recursive(fnus, sort_indices),
            "ObservedSpectra",
            units=units["fnu"],
        )

        # Write the photometry
        write_dataset_recursive(
            hdf,
            sort_data_recursive(fluxes, sort_indices),
            key="ObservedPhotometry",
            units=units["flux"],
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(rf_fluxes, sort_indices),
            key="RestFramePhotometry",
            units=units["flux"],
        )

        # Write the UV slopes
        write_dataset_recursive(
            hdf,
            sort_data_recursive(uv_slopes, sort_indices),
            key="UVSlopes",
            units="dimensionless",
        )

        # Write the IR slopes
        write_dataset_recursive(
            hdf,
            sort_data_recursive(ir_slopes, sort_indices),
            key="IRSlopes",
            units="dimensionless",
        )

        # Write the light radii
        write_dataset_recursive(
            hdf,
            sort_data_recursive(sizes, sort_indices),
            key="HalfLightRadii",
            units=units["hlr"],
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(sizes_95, sort_indices),
            key="LightRadii95",
            units=units["hlr"],
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(sizes_80, sort_indices),
            key="LightRadii80",
            units=units["hlr"],
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(sizes_20, sort_indices),
            key="LightRadii20",
            units=units["hlr"],
        )

        # Write the images
        write_dataset_recursive(
            hdf,
            sort_data_recursive(imgs, sort_indices),
            key="Images",
            units=units["flux"],
        )

        # Write the apertures
        write_dataset_recursive(
            hdf,
            sort_data_recursive(apps, sort_indices),
            key="Apertures",
            units=units["flux"],
        )

        # Write out the indices
        write_dataset_recursive(
            hdf,
            sort_data_recursive(indices, sort_indices),
            "Indices",
        )

        # Write out the gas sizes
        write_dataset_recursive(
            hdf,
            sort_data_recursive(gas_sizes, sort_indices),
            key="GasHalfMassRadius",
            units=units["hlr"],
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(gas_sizes_80, sort_indices),
            key="GasMassRadius80",
            units=units["hlr"],
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(gas_sizes_20, sort_indices),
            key="GasMassRadius20",
            units=units["hlr"],
        )
        write_dataset_recursive(
            hdf,
            sort_data_recursive(dust_sizes, sort_indices),
            key="DustHalfMassRadius",
            units=units["hlr"],
        )

        # Store the sfzhs
        write_dataset_recursive(
            hdf,
            sort_data_recursive(sfzhs, sort_indices),
            "SFZH",
            units=units["sfzh"],
        )

        # Write the image fluxes
        write_dataset_recursive(
            hdf,
            sort_data_recursive(img_fluxes, sort_indices),
            key="ImageObservedPhotometry",
            units=units["flux"],
        )


# Define the snapshot tags
snapshots = [
    "005_z010p000",
    "006_z009p000",
    "007_z008p000",
    "008_z007p000",
    "009_z006p000",
    "010_z005p000",
]

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Derive synthetic observations for FLARES."
    )

    # Add the arguments
    parser.add_argument(
        "master_file_path",
        type=str,
        help="The path to the master file.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        help="The path to the grid.",
    )
    parser.add_argument(
        "--grid-dir",
        type=str,
        help="The directory to save the grid.",
    )
    parser.add_argument(
        "--region",
        type=int,
        help="The region to use.",
    )
    parser.add_argument(
        "--snap",
        type=int,
        help="The snapshot to use.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use.",
    )

    # Get MPI info
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse the arguments
    args = parser.parse_args()
    path = args.master_file_path
    grid_name = args.grid
    grid_dir = args.grid_dir
    region = args.region
    snap = snapshots[args.snap]
    nthreads = args.nthreads

    # Define the output path
    outpath = f"data/RUBIES_COMP_{str(region).zfill(2)}_{snap}.hdf5"

    # If the output already exists just exit
    if os.path.exists(outpath):
        _print(f"Output file {outpath} already exists.")
        exit(0)

    start = time.time()

    # Get the filters
    filt_start = time.time()
    filters = get_flares_filters("rubies_filters.hdf5")
    filt_end = time.time()
    _print(f"Getting filters took {filt_end - filt_start:.2f} seconds.")

    # Get the grid
    grid_start = time.time()
    grid, filters = get_grid(grid_name, grid_dir, filters)
    grid_end = time.time()

    # Get the PSFs
    psf_start = time.time()
    psfs = get_psfs(FILTER_CODES, "rubies_psfs.hdf5")
    psf_end = time.time()
    _print(f"Getting the PSFs took {psf_end - psf_start:.2f} seconds.")

    # Get the emission model
    start_emission = time.time()
    emission_model = get_emission_model(grid)
    end_emission = time.time()
    _print(
        f"Getting the emission model took "
        f"{end_emission - start_emission:.2f} seconds."
    )

    # If we're on rank 0 and the first region and snapshot plot the model
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0 and region == 0 and snap == snapshots[0]:
        fig, ax = emission_model.plot_emission_tree(fontsize=8)
        fig.savefig("plots/emission_tree.png", dpi=300, bbox_inches="tight")

    # Get the kernel
    start_kernel = time.time()
    kern = get_kernel()
    end_kernel = time.time()

    # Get the galaxies
    read_start = time.time()
    galaxies = get_flares_galaxies(
        path,
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

    # Analyse the galaxy
    gal_start = time.time()
    galaxies = [
        analyse_galaxy(
            gal,
            emission_model,
            grid,
            kern,
            nthreads,
            filters,
            cosmo,
            psfs,
        )
        for gal in galaxies
    ]
    gal_end = time.time()
    _print(
        f"Analysing {len(galaxies)} galaxies took "
        f"{gal_end - gal_start:.2f} seconds."
    )

    # Write out the results
    write_start = time.time()
    write_results(
        galaxies,
        outpath,
        grid_name,
        filters,
        comm,
        rank,
        size,
    )

    comm.barrier()

    end = time.time()
    _print(f"Total time: {end - start:.2f} seconds.")
