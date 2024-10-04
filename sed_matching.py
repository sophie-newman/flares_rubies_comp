import numpy as np
from astropy.table import Table
from utils import read_sed_from_hdf5, resample_fluxes, normalise_seds, find_matches

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

# FLARES region numbers.
regions = ['00','01','02','03','04','05','06','07','08','09'] + [int(i) for i in np.arange(10, 40)]

# Store the FLARES SEDs, region numbers and master file indices.
flares_seds = 0
region_num = []
indices = []

# For each FLARES region.
for region in regions:

    # Get the path,excluding region 38.
    path = f'/research/astro/flare/data/flares_rubies_comp/RUBIES_COMP_{region}_008_z007p000.hdf5'
    if '38' in path:
        continue

    # Get the SED and master file indices.
    seds, indices_ = read_sed_from_hdf5(path, filters, 'ObservedPhotometry/total/JWST', 
                                        conversion=(1e23*1e9))

    # Store region numbers and indices.
    region_num += [region]*len(indices_)
    indices += [int(i) for i in indices_]

    # Add additional SEDs to the array.
    if isinstance(flares_seds, int):
        flares_seds = seds
    else:
        flares_seds = np.vstack((flares_seds, seds))

region_num = np.array(region_num)
indices = np.array(indices)

# The scalers to use.
scalers = ['Standard', 'MinMax', 0, 1, 2, 3, 4, 5, 6, 7, 8]
# The number of resampling iterations.
its = 1000

# Will store the matching information of each source in a Table.
match_table = Table()
match_table['region'] = region_num
match_table['indices'] = indices

# For each type of scaler.
for scaler in scalers:

    print(f'Scaler: {scaler}')

    # Get an array of resampled and normalised RUBIES SEDS.
    rubies_seds = resample_fluxes(rubies, its)
    rubies_seds = normalise_seds(rubies_seds, scaler=scaler)

    # Normalise the FLARES galaxies in the same way.
    norm_flares = normalise_seds(flares_seds, scaler=scaler)

    # Store the matching info here.
    match_info = np.zeros(shape=(norm_flares.shape[0], 2))

    for sed in rubies_seds:

        # Find the best matching galaxies by shape.
        best, distances = find_matches(sed, norm_flares)

        # Record how many times an SED has had the best match
        match_info[best, 1] += 1

        # and the total distance.
        match_info[:, 0] += distances

    # Store the total number of best matches and total distance for each
    # object using this scaler.
    if isinstance(scaler, str):
        label = scaler
    else:
        label = filters[scaler]

    match_table[f'{label}_Nbest'] = match_info[:, 1]
    match_table[f'{label}_distance'] = match_info[:, 0]

# Find all _Nbest and _distance columns.
nbest_columns = [col for col in match_info.colnames if col.endswith('_Nbest')]
distance_columns = [col for col in match_info.colnames if col.endswith('_distance')]

# Create new 'total' columns for each.
match_info['total_Nbest'] = np.sum([match_info[col] for col in nbest_columns], axis=0)
match_info['total_distance'] = np.sum([match_info[col] for col in distance_columns], axis=0)

# Create a sub sample of galaxies that matched at least once.
s = match_info['total_Nbest'] > 0
fr_sample = match_info[s]

match_table.write('/research/astrodata/highz/flares_rubies/matching_table.fits', overwrite=True)
fr_sample.write('/research/astrodata/highz/flares_rubies/FR_sample.fits', overwrite=True)