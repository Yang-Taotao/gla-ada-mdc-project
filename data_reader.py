"""
This is the data reader module file for MDC data reading.

Created on Mon Mar 13 15:20:15 2023

@author: Yang-Taotao
"""
# %% Library import
# Library import
import numpy as np

# %% Data reader function
# Data reader
def file_loader(file_path):
    """
    Parameters
    ----------
    file_path : string
        Path to data file

    Returns
    ----------
    data_x : numpy array
        An array of x data
    data_y : numpy array
        An array of y data
    """
    data = np.loadtxt(file_path)
    data_x = data[:, 0]
    data_y = data[:, 1]
    return data_x, data_y
