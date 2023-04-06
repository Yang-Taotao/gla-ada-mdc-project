"""
This is the master script for MDC of ADA course.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Library import
# Library import
import numpy as np
import corner
import matplotlib.pyplot as plt
# Local module import
from data_reader import file_loader
from data_fitter import linear_ls, linear_ml

# %% Data loader for MDC1 and MDC2
# Assign file path
file_1, file_2 = "./data/MDC1.txt", "./data/MDC2.txt"
# Assign x,y data array from function call
data_x1, data_y1 = file_loader(file_1)
data_x2, data_y2 = file_loader(file_2)

# %% 1.1 - Get ordinary linear least squares fit result for MDC1.txt
fit_ls_result = linear_ls(data_x1, data_y1)

# %% 1.2 - Get maximum likelihood fitting result for MDC1.txt
linear_ml(data_x1, data_y1, fit_ls_result[0], fit_ls_result[1])
