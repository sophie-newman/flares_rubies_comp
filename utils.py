import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr
import h5py

def resample_fluxes(sed, n):
    """
    Resample fluxes from a Gaussian based on their uncertainties..
    
    Arguments
    ---------
    sed (numpy.ndarray)
        2D array describing an SED. First column stores fluxes the and
        the second errors.
    n (int)
        The number of resampling iterations.
        
    Returns
    -------
    resampled (numpy.ndarray)
        Array of resampled fluxes with shape (n, len(sed[:, 0]))    
    """

    # Generate resampled fluxes in a single vectorized operation
    resampled = np.random.normal(loc=sed[:, 0], scale=sed[:, 1], size=(n, len(sed[:, 0])))

    return resampled

def normalise_seds(seds, scaler='MinMax'):
    """
    Normalise a 2D array of SEDs with shape.
    
    Arguments
    ---------
    seds (numpy.ndarray)
        2D array of SEDs to normalise. Each row is assumed to be an SED.
    scaler (str/int)
        The type of scaler to use. If an integer is provided, normalise 
        by the flux at that SED index.
    
    Returns
    -------
    norm_seds (numpy.ndarray)
        2D array of normalised SEDs.
    """

    # If a 1D array is provided, reshape it to 2D.
    orig_shape = seds.shape
    if seds.ndim == 1:
        seds = seds.reshape(1, -1)

    # Scale using the requested approach.
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
        norm_seds = scaler.fit_transform(seds.T).T

    elif scaler == 'Standard':
        scaler = StandardScaler()
        norm_seds = scaler.fit_transform(seds.T).T

    elif scaler == 'Sum':
        sed_sums = np.sum(seds, axis=1)
        norm_seds = seds / sed_sums[:, np.newaxis]

    # Normalise by the flux at the provided index.
    elif isinstance(scaler, int):
        norm_seds = seds / seds[:, scaler].reshape(-1, 1)
    
    else:
        raise KeyError(f'{scaler} is not a valid scaler.')
    
    # If original input was (N,), convert back.
    if len(orig_shape) == 1:
        return norm_seds.flatten() 
    
    return norm_seds  

def find_matches(reference, seds):
    """
    Use Pearson correlation to match SEDs to a reference.
    
    Arguments
    ---------
    reference (numpy.ndarray)
        1D array of normalised reference SED fluxes.
    seds (numpy.ndarray)
        2D array of normalised SEDs to match to the reference. 
        Assumes each row corresponds to an SED.

    Returns
    -------
    match_index (int)
        Index into distances of closest matching SED.
    distances (numpy.ndarray)
        1D array of distances based on the Pearson coefficient.
    """
    
    # Store distances here.
    distances = []

    # For each  SED.
    for sed in seds:

        # Calculate the Pearson correlation coefficient.
        corr_coeff, _ = pearsonr(reference, sed)

        # Distance is probability of not being correlated.
        dist = 1 - corr_coeff 
        distances.append(dist)
    
    # Return the index of the best matching source and the distances.
    match_index = np.argmin(distances)
    distances = np.array(distances)

    return match_index, distances

def read_sed_from_hdf5(path, filters, phot_path='', conversion=1):
    """
    Move SEDs stored in an hdf5 file to an array.
    
    Arguments
    ---------
    path (str)
        Path to the hdf5 file.
    filters (list[str])
        List of filters to include in the SED.
    phot_path (str)
        Initial path element to reach the filters group.
    conversion (float)
        Multiplicative conversion from catalogue units to output units.
        
    Returns
    -------
    seds (numpy.ndarray)
        2D array where each row corresponds to a single SED.
    """

    # Store the fluxes here.
    seds = []

    # Read the hdf5 file.
    with h5py.File(path, 'r') as f:

        # Add the flux in each band to the SED.
        for filter in filters:
            seds.append(f[f'{phot_path}/{filter}'][:]*conversion)

        # Also return the master file index.
        indicies = f['Indices'][:]

    # Get an array of SEDs in the correct format.
    seds = np.array(seds).T

    return seds, indicies