"""Script to plot SFH comparisons for RUBIES-like FLARES galaxies."""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo

# Defines the regions and indices for the RUBIES-like FLARES galaxies
regions = []
indices = []

# Define the best match
best_reg = "00"
best_ind = 0

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
bins = np.linspace(0, cosmo.age(7.0).to("Myr").value, 10)

# Define the rubies bins
bins = np.linspace(0, cosmo.age(7.0).to("Myr").value, 100)

# Set up the plot
fig = plt.figure(figsize=(2 * 3.5, 2 * 3.5))

# Add a grid spec
gs = fig.add_gridspec(2, 1)
ax_flares = fig.add_subplot(gs[0])
ax_rubies = fig.add_subplot(gs[1])

# Add a grid and make sure it's behind everything
ax_flares.grid(True)
ax_flares.set_axisbelow(True)
ax_rubies.grid(True)
ax_rubies.set_axisbelow(True)

# Loop over the "non-best" galaxies plotting their SFHs in low alpha
for (reg, ind), mass, age in zip(
    masses.keys(), masses.values(), ages.values()
):
    ax_flares.hist(
        age,
        bins=bins,
        weights=mass / 10 * 10**6,
        color="black",
        alpha=0.4,
    )

# Plot the best match in high alpha
ax_flares.hist(
    best_ages,
    bins=bins,
    weights=best_masses / 10 * 10**6,
    color="red",
)

# Now do the same using the rubies binning
for (reg, ind), mass, age in zip(
    masses.keys(), masses.values(), ages.values()
):
    ax_rubies.hist(
        age,
        bins=bins,
        weights=mass / 100 * 10**6,
        histtype="step",
        color="black",
        alpha=0.4,
    )

# Plot the best match in high alpha
ax_rubies.hist(
    best_ages,
    bins=bins,
    weights=best_masses / 100 * 10**6,
    histtype="step",
    color="red",
)

# Remove the upper x-axis labels
ax_flares.set_xticklabels([])

# Set the labels
ax_flares.set_ylabel("SFR $/ [\mathrm{M}_\odot / \mathrm{yr}]$")
ax_rubies.set_ylabel("SFR $/ [\mathrm{M}_\odot / \mathrm{yr}]$")
ax_rubies.set_xlabel("Age $/ [\mathrm{Myr}]$")

# Save the figure
fig.savefig("rubies_sfh.pdf", bbox_inches="tight")
