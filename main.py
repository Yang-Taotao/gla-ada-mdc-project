"""
This is the master script for MDC of ADA course.
Tuple unpacking and indexing is heavily used for this project.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Module import
# Custom data reader module
from data_reader import file_loader

# Custom data fitter module
from data_fitter import linear_ls, linear_ml

# %% Data loader for MDC1 and MDC2
# Assign file path
file_1, file_2 = "./data/MDC1.txt", "./data/MDC2.txt"
# Assign x,y data array from function call
data_x1, data_y1 = file_loader(file_1)  # MDC1 dataset
data_x2, data_y2 = file_loader(file_2)  # MDC2 dataset

# %% 1.1 - Get ordinary linear least squares fit result print out for MDC1.txt
linear_ls(data_x1, data_y1)

# %% 1.2 - Get maximum likelihood fitting result for MDC1.txt
linear_ml(data_x1, data_y1)
