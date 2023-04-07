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
from data_fitter import linear_ls, linear_ml, linear_mcmc

# Custom data plotter module
from data_plotter import baysian_plotter

# %% Data loader for MDC1 and MDC2
# Assign file path
file_1, file_2 = "./data/MDC1.txt", "./data/MDC2.txt"
# Assign x,y data array from function call
data_x1, data_y1 = file_loader(file_1)  # MDC1 dataset
data_x2, data_y2 = file_loader(file_2)  # MDC2 dataset

# %% 1.1 - Get ordinary linear least squares fit result print out for MDC1.txt
res_linear_ls = linear_ls(data_x1, data_y1)

# %% 1.2 - Get maximum likelihood fitting result for MDC1.txt
# Deposit linear ls fit param for ml initial guesses
res_linear_ml = linear_ml(data_x1, data_y1, res_linear_ls)
# Generate ml baysian credible region plot
baysian_plotter(res_linear_ml)

# %% 1.3 - MCMC linear fit with corner plot
linear_mcmc(data_x1, data_y1)