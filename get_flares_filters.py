import os
import numpy as np
from unyt import angstrom
from synthesizer.instruments import UVJ, FilterCollection, Instrument

from utils import RUBIES_FILTER_CODES as FILTER_CODES

def get_flares_filters(filepath):
    """Get the filter collection."""
    # Check if the filter collection file already exists
    if os.path.exists(filepath):
        filters = FilterCollection(path=filepath)
    else:
        # UVJ
        tophats = {
        "V": {"lam_eff": 5510 * angstrom, "lam_fwhm": 880 * angstrom}
    }

        # Create the FilterCollection
        filters = FilterCollection(
            filter_codes=FILTER_CODES[1:],
            tophat_dict=tophats,
        )

        # Write the filter collection
        filters.write_filters(path=filepath)

    return filters

flares_filters = get_flares_filters("rubies_filters.hdf5")