import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from utils import read_sed_from_hdf5, normalise_seds
plt.style.use('/its/home/jt458/style.mplstyle')

# List of filters to use.
filters = [f'NIRCam.{band}' for band in ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 
                                         'F410M', 'F444W']] + ['MIRI.F770W']

# and their pivot wavelength,
pivots = np.array([9021.53, 11542.61, 15007.44, 19886.48, 27617.40, 35683.62,
                    40822.38, 44043.15, 76393.34])
# in microns.
pivots /= 10000

# The total RUBIES fluxes in nJy.
rubies_fluxes = [-3.7, 45.2, 75.7, 108.3, 186.0, 469.6, 522.5, 527.6, 673.7]
rubies_errors = [7.7, 7.9, 6.7, 5.6, 9.3, 23.5, 26.1, 26.4, 94.8]
rubies = np.column_stack((rubies_fluxes, rubies_errors))

# Start by plotting the absolute SEDs.
plt.figure()
plt.errorbar(pivots, rubies[:, 0], rubies[:, 1], color = 'black')
plt.xlabel('$\\lambda [\\mu\\mathrm{n}]$')
plt.ylabel('F$_{\\nu}$ [nJy]')

# FLARES region numbers.
regions = ['00','01','02','03','04','05','06','07','08','09'] + [int(i) for i in np.arange(10, 40)]

# Store the FLARES SEDs..
flares_seds = 0

# For each FLARES region.
for region in regions:

    # Get the path,excluding region 38.
    path = f'/research/astro/flare/data/flares_rubies_comp/RUBIES_COMP_{region}_008_z007p000.hdf5'
    if '38' in path:
        continue

    # Get the SED and master file indices.
    seds, _ = read_sed_from_hdf5(path, filters, 'ImageObservedPhotometry/attenuated/JWST', 
                                        conversion=(1e23*1e9))

    # Add additional SEDs to the array.
    if isinstance(flares_seds, int):
        flares_seds = seds
    else:
        flares_seds = np.vstack((flares_seds, seds))


plt.plot(pivots, flares_seds.T, color = 'grey', alpha=0.05)

plt.savefig('plots/absolute_sed_comp.png', dpi=300, bbox_inches='tight')

# Now compare the shapes using different normalisations.
plt.figure()

plt.xlabel('$\\lambda [\\mu\\mathrm{n}]$')
plt.ylabel('Normalised F$_{\\nu}$')

scalers = ['Standard', 'MinMax', 0, 1, 2, 3, 4, 5, 6, 7, 8]

# Then for each scaler.
for scaler in scalers:

    plt.figure()

    plt.xlabel('$\\lambda [\\mu\\mathrm{n}]$')
    plt.ylabel('Normalised F$_{\\nu}$')

    # Also normalise the RUBIES SED.
    norm_rubies = normalise_seds(rubies[:, 0], scaler=scaler)

    # Normalise the FLARES galaxies in the same way.
    norm_flares = normalise_seds(flares_seds, scaler=scaler)

    # Plot both.
    plt.plot(pivots, norm_rubies, color = 'black')
    plt.plot(pivots, norm_flares.T, color='grey', alpha=0.05)

    if isinstance(scaler, str):
        label = scaler
    else:
        label = filters[scaler]

    plt.savefig(f'plots/{label}_normalised_seds.png', dpi=300, bbox_inches='tight')

# Colour-colour plot.

# Colour is calculated as:
# A - B = -2.5 * log10(A/B)

col1_short, col1_long = 2, 3    # Probe the Balmer break.
col2_short, col2_long = 4, 5    # Something to plot agaianst.

# Calculate the colours for each FLARES SED.
color1 = -2.5*np.log10(flares_seds[:, col1_short]/flares_seds[:, col1_long])
color2 = -2.5*np.log10(flares_seds[:, col2_short]/flares_seds[:, col2_long])

# Calculate the colour and error for the RUBIES galaxy.
col1_rub = -2.5*np.log10(rubies[col1_short, 0]/rubies[col1_long, 0])
col1_rub_err = np.sqrt(((rubies[col1_short, 1]/rubies[col1_short, 0])**2) + ((rubies[col1_long, 1]/rubies[col1_long, 0])**2))*1.0857

col2_rub = -2.5*np.log10(rubies[col2_short, 0]/rubies[col2_long, 0])
col2_rub_err = np.sqrt(((rubies[col2_short, 1]/rubies[col2_short, 0])**2) + ((rubies[col2_long, 1]/rubies[col2_long, 0])**2))*1.0857

plt.figure(figsize=(8, 6))
plt.scatter(color1, color2, c='grey', marker='o', zorder=0, s = 2)
plt.errorbar(col1_rub, col2_rub, yerr=col2_rub_err, xerr=col1_rub_err, c='red', zorder=1, fmt='.', linewidth=1, capsize=2)
plt.xlabel('F150W - F200W')
plt.ylabel('F277W - F356W')
plt.savefig('plots/balmer_cc.png', dpi=300, bbox_inches='tight')