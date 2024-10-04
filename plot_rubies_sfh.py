"""Script to plot SFH comparisons for RUBIES-like FLARES galaxies."""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits

# Define the path to the RUBIES matching data
rubies_matches = "FR_sample.fits"

# Open the fits file
with fits.open(rubies_matches) as hdul:
    # Get the data
    data = hdul[1].data

    # Defines the regions and indices for the RUBIES-like FLARES galaxies
    regions = data["region"]
    indices = data["indices"]

    # Define the best match
    best_reg = regions[np.argmin(data["total_distance"])]
    best_ind = indices[np.argmin(data["total_distance"])]

# They're all in the same snapshot
snap = "008_z007p000"

# Define the master file path
master_path = (
    "/cosma7/data/dp004/dc-payy1/my_files//flares_pipeline/data/flares.hdf5"
)

# Set up dicts to contain the masses and ages
masses = {}
ages = {}

# Open the master file
with h5py.File(master_path, "r") as hdf:
    # Loop over the galaxies
    for reg, ind in zip(regions, indices):
        # Define the key
        key = f"{reg}/{snap}/"

        # Get the stellar length array
        slength = hdf[f"{key}/Galaxy/S_Length"][:]

        # Get the start indices
        sstart = np.cumsum(slength)
        sstart = np.insert(sstart, 0, 0)

        # Get the particle stellar masses and ages
        smass = (
            hdf[f"{key}/Particle/S_Mass"][sstart[ind] : sstart[ind + 1]]
            * 10**10
        )
        sage = (
            hdf[f"{key}/Particle/S_Age"][sstart[ind] : sstart[ind + 1]] * 1000
        )

        # Add to the dicts
        masses[(reg, ind)] = smass
        ages[(reg, ind)] = sage

    # Do the same for the best match
    key = f"{best_reg}/{snap}/"

    # Get the stellar length array
    slength = hdf[f"{key}/Galaxy/S_Length"][:]
    sstart = np.cumsum(slength)
    sstart = np.insert(sstart, 0, 0)

    # Get the particle stellar masses and ages
    best_masses = (
        hdf[f"{key}/Particle/S_Mass"][sstart[best_ind] : sstart[best_ind + 1]]
        * 10**10
    )
    best_ages = (
        hdf[f"{key}/Particle/S_Age"][sstart[best_ind] : sstart[best_ind + 1]]
        * 1000
    )

# Define the bins for the high resolution SFH
flares_bins = np.arange(0, cosmo.age(7.0).to("Myr").value, 10)
flares_bin_centers = (flares_bins[1:] + flares_bins[:-1]) / 2

# Define the rubies bins
rubies_bins = np.linspace(0, cosmo.age(7.0).to("Myr").value - 100, 5)
rubies_bins = np.hstack(
    (
        rubies_bins,
        np.array(
            [
                cosmo.age(7.0).to("Myr").value - 50,
                cosmo.age(7.0).to("Myr").value - 10,
                cosmo.age(7.0).to("Myr").value,
            ]
        ),
    )
)
rubies_bin_centers = (rubies_bins[1:] + rubies_bins[:-1]) / 2

# Set up the plot
fig = plt.figure(figsize=(2 * 3.5, 2 * 3.5))

# Add a grid spec
gs = fig.add_gridspec(2, 1, hspace=0)
ax_flares = fig.add_subplot(gs[0])
ax_rubies = fig.add_subplot(gs[1])

# Add a grid and make sure it's behind everything
ax_flares.grid(True)
ax_flares.set_axisbelow(True)
ax_rubies.grid(True)
ax_rubies.set_axisbelow(True)

# Log scale both y axes
ax_flares.set_yscale("log")
ax_rubies.set_yscale("log")

# Loop over the "non-best" galaxies plotting their SFHs in low alpha
for (reg, ind), mass, age in zip(
    masses.keys(), masses.values(), ages.values()
):
    # Histogram the masses with numpy
    H, _ = np.histogram(age, bins=flares_bins, weights=mass)

    # Plot the histogram
    ax_flares.plot(
        flares_bin_centers,
        H / 10 / 10**6,
        color="black",
        alpha=0.2,
    )

# Plot the best match in high alpha
H, _ = np.histogram(best_ages, bins=flares_bins, weights=best_masses)
ax_flares.plot(flares_bin_centers, H / 10 / 10**6, color="red")

# Now do the same using the rubies binning but use steps here
for (reg, ind), mass, age in zip(
    masses.keys(), masses.values(), ages.values()
):
    # Histogram the masses with numpy
    H, _ = np.histogram(age, bins=rubies_bins, weights=mass)

    # Plot the histogram but using steps
    ax_rubies.plot(
        rubies_bin_centers,
        H / 100 / 10**6,
        color="black",
        alpha=0.2,
        drawstyle="steps",
    )


# Plot the best match in high alpha
H, _ = np.histogram(best_ages, bins=rubies_bins, weights=best_masses)
ax_rubies.plot(
    rubies_bin_centers,
    H / 100 / 10**6,
    color="red",
    drawstyle="steps",
)

# Remove the upper x-axis labels
ax_flares.set_xticklabels([])

# Apply the same limits
ax_rubies.set_xlim(150, 750)
ax_flares.set_xlim(ax_rubies.get_xlim())
ax_rubies.set_ylim(ax_flares.get_ylim())

# # Reverse the x-axes
# ax_flares.invert_xaxis()
# ax_rubies.invert_xaxis()

# Set the labels
ax_flares.set_ylabel("SFR $/ [\mathrm{M}_\odot / \mathrm{yr}]$")
ax_rubies.set_ylabel("SFR $/ [\mathrm{M}_\odot / \mathrm{yr}]$")
ax_rubies.set_xlabel("Age $/ [\mathrm{Myr}]$")

# Save the figure
fig.savefig("rubies_sfh.png", bbox_inches="tight", dpi=100)
