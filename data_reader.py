"""
This is the data reader module file for MDC data reading.

Created on Mon Mar 13 15:20:15 2023

@author: Yang-Taotao
"""
# %% Library import
# Import numpy with np alias for array manupulations
import numpy as np

# %% Data reader function
# Data reader
def file_loader(file_path):
    """
    Parameters
    ----------
    file_path : string
        Path to data file.

    Returns
    ----------
    data_x : array
        An array of x data.
    data_y : array
        An array of y data.
    """
    # Read txt file to numpy array with np.loadtext()
    data = np.loadtxt(file_path)
    # Unpack loaded data to x, y arrays and pack it back into result output
    result = (data[:, 0], data[:, 1])
    # Return function call
    return result
