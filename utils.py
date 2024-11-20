import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr
import h5py
from mpi4py import MPI as mpi

# Define spectra keys
SPECTRA_KEYS = [
    "old_transmitted",
    "young_transmitted",
    "old_nebular",
    "young_nebular",
    "old_reprocessed",
    "young_reprocessed",
    "old_escaped",
    "young_escaped",
    "young_attenuated",
    "old_attenuated",
    "young_intrinsic",
    "old_intrinsic",
    "stellar_intrinsic",
    "agn_intrinsic",
    "escaped",
    "stellar_attenuated",
    "agn_attenuated",
    "stellar_total",
    "combined_intrinsic",
    "total_dust_free_agn",
    "total",
    "young_attenuated_nebular",
    "incident",
    "transmitted",
    "reprocessed",
    "old_incident",
    "young_incident",
    "nebular"
    
]

RUBIES_FILTER_CODES = [
    "UV1500",
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F140M",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F162M",
    "JWST/NIRCam.F182M",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F210M",
    "JWST/NIRCam.F250M",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F300M",
    "JWST/NIRCam.F335M",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F360M",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F430M",
    "JWST/NIRCam.F444W",
    "JWST/NIRCam.F460M",
    "JWST/NIRCam.F480M",
    "JWST/MIRI.F770W",
    "JWST/MIRI.F1800W",
]

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

def write_dataset_recursive(hdf, data, key, units="dimensionless"):
    """
    Write a dictionary to an HDF5 file recursively.

    Args:
        hdf (h5py.File): The HDF5 file to write to.
        data (dict): The data to write.
        key (str): The key to write the data to.
        units (str): The units of the data.
    """
    # If the data isn't a dictionary just write the dataset
    if not isinstance(data, dict):
        _print(f"Writing {key}")
        dset = hdf.create_dataset(key, data=data)
        dset.attrs["Units"] = units
        return

    # Loop over the data
    for k, v in data.items():
        write_dataset_recursive(hdf, v, f"{key}/{k}", units=units)

def sort_data_recursive(data, sinds):
    """
    Sort a dictionary recursively.

    Args:
        data (dict): The data to sort.
        sinds (dict): The sorted indices.
    """
    # If the data isn't a dictionary just return the sorted data
    if not isinstance(data, dict):
        data = np.array(data)
        return data[sinds]

    # Loop over the data
    sorted_data = {}
    for k, v in data.items():
        sorted_data[k] = sort_data_recursive(v, sinds)

    return sorted_data

def _print(*args, **kwargs):
    """Overload print with rank info."""
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"[{str(rank).zfill(len(str(size)) + 1)}]: ", end="")
    print(*args, **kwargs)


