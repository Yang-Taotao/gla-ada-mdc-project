"""
## Description: 
- This is the master script for MDC of ADA course.
- Tuple unpacking and indexing is heavily used for this project.
- Results are generated with customized function calls.
- Printouts are formatted with f-strings methods.

## Method:
- This script first import the necessary modules.
- Then defines the file path and model strings.
- It then assigns the data array from txt files.
- Afterwards, the results required for each section is called.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Module import
# Custom data reader module
# Module file_loader loads the txt file into numpy arrays
from data_reader import file_loader

# Custom data fitter module import
# Module linear_ls performs linear least squares fitting
# Module linear_ml performs linear maximum likelihood fitting
from data_fitter_linear import linear_ls, linear_ml

# Module linear_mcmc performs MCMC on a linear model
from data_fitter_mcmc import mcmc_fitter

# Module bayes_factor calculates the bayes factor
from data_fitter_bayes import bayes_factor

# Custom data plotter module
# Module baysian_plotter generates the plot for 1.2
# Module linear_mcmc_plotter generates the plot for 1.3 and 2.1
from data_plotter import baysian_plotter, mcmc_plotter

# %% Data loader for MDC1 and MDC2
# Assign file path and associated model as file_path and file_model
file_path_1, file_path_2, file_model_1, file_model_2 = (
    "./data/MDC1.txt",  # File path for MDC1 data
    "./data/MDC2.txt",  # File path for MDC2 data
    "Linear",  # Linear model string
    "Quadratic",  # Quadraitc model string
)

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
# Initial guesses made with all fit param = 1.0
# After the initial fit, new param are updated for better plotting
# Deposit mcmc data with linear model
res_line_mcmc = mcmc_fitter(data_x1, data_y1, file_model_1)
# Generate mcmc corner plot
mcmc_plotter(res_line_mcmc, file_model_1)

# %% 2.1 - MCMC quadratic fit with corner plot
# Initial guesses made with all fit param = 1.0
# After the initial fit, new param are updated for better plotting
# Deposit mcmc data with quadratic model
res_quad_mcmc = mcmc_fitter(data_x2, data_y2, file_model_2)
# Generate mcmc corner plot
mcmc_plotter(res_quad_mcmc, file_model_2)

# %% 2.2 - Bayes factor calculation and result printout
bayes_factor(data_x1, data_y1, data_x2, data_y2)
