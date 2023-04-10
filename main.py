"""
This is the master script for MDC of ADA course.
Tuple unpacking and indexing is heavily used for this project.
Results are generated with customized function calls.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Module import
# Custom data reader module
# Module file_loader loads the txt file into numpy arrays
from data_reader import file_loader

# Custom data fitter module
# Module linear_ls performs linear least squares fitting
# Module linear_ml performs linear maximum likelihood fitting
# Module linear_mcmc performs MCMC on a linear model
from data_fitter import linear_ls, linear_ml, mcmc_fitter

# Custom data plotter module
# Module baysian_plotter generates the plot for 1.2
# Module linear_mcmc_plotter generates the plot for 1.3
from data_plotter import baysian_plotter, linear_mcmc_plotter

# %% Data loader for MDC1 and MDC2
# Assign file path as file_path
file_path_1, file_path_2 = "./data/MDC1.txt", "./data/MDC2.txt"
# Assign x,y data array from function call
(data_x1, data_y1), (data_x2, data_y2) = (
    file_loader(file_path_1),  # MDC1 dataset
    file_loader(file_path_2),  # MDC2 dataset
)

# %% 1.1 - Get ordinary linear least squares fit result print out for MDC1.txt
res_linear_ls = linear_ls(data_x1, data_y1)

# %% 1.2 - Get maximum likelihood fitting result for MDC1.txt
# Deposit ml data with linear ls fit result as initial guesses
res_linear_ml = linear_ml(data_x1, data_y1, res_linear_ls[0])
# Generate ml baysian credible region plot
baysian_plotter(res_linear_ml)

# %% 1.3 - MCMC linear fit with corner plot
# Deposit mcmc data with linear ls fit result as intial guesses
res_linear_mcmc = mcmc_fitter(data_x1, data_y1, res_linear_ls)
# Generate mcmc corner plot
linear_mcmc_plotter(res_linear_mcmc)
