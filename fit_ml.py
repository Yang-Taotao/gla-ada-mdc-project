"""
This is the module file for performaing maximum likelihood fitting.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# Library import
import numpy as np
import matplotlib.pyplot as plt

# Maximum likelihood linear model fitting
def linear_ml(data_x, data_y, aml, bml):
    # Define the likelihood function with assumed Gaussian distribution
    def linear_log_likelihood(data_x, data_y, aml, bml):
        """
        Parameters
        ----------
        data_x : numpy array
            Data array of x
        data_y : numpy array
            Data array of y
        aml : float
            MLE intercept value
        bml : float
            MLE slope value

        Returns
        -------
        log_likeihood_val : float
            Log likelihood value
        """
        # Calculate the standard deviation of y entries and sample size
        sigma, data_n = np.std(data_y), len(data_x)
        # Define linear fit model
        model_ml = aml + bml * data_x
        # Generate mean value of the model
        model_mean = np.mean(model_ml)

        # Generate likelihood function with the model mean in product format
        likelihood_val = np.prod(
            (np.exp(-((data_y - model_mean) ** 2) / (2 * sigma**2)))
            / (np.sqrt(2 * np.pi) * sigma)
        )
        # Generate log likelihood function with the model mean as sums
        log_likelihood_val = (
            -0.5 * data_n * np.log(2 * np.pi * sigma**2)
        ) - (1 / (2 * sigma**2)) * np.sum((data_y - model_mean) ** 2)
        # Return log likelihood function value calculation result
        return log_likelihood_val

    # Define chi-square calculation
    def linear_chi_sqr(data_x, data_y, aml, bml):
        """
        Parameters
        ----------
        data_x : numpy array
            Data array of x
        data_y : numpy array
            Data array of y
        aml : float
            MLE intercept value
        bml : float
            MLE slope value

        Returns
        -------
        chi_sqr_val : float
            Chi-squared value
        """
        # Calculate the standard deviation of y entries and sample size
        sigma, data_n = np.std(data_y), len(data_x)
        # Define linear fit model
        model_ml = aml + bml * data_x
        # Generate mean value of the model
        model_mean = np.mean(model_ml)

        # Perform chi-squared value calculation
        chi_sqr_val = np.sum(((data_y - model_mean) / (sigma)) ** 2)
        return chi_sqr_val

    # Make initial slope and intercept iteration array from intial guess
    aml_val, bml_val = (
        np.linspace(aml - 5, aml + 5, 1000),
        np.linspace(bml - 5, bml + 5, 1000),
    )

    # Generate parameter pair grid
    aml_grid, bml_grid = np.meshgrid(aml_val, bml_val)

    # Generate log likelihood value matrix
    # log_likelihood_mat = linear_log_likelihood(
    #     data_x, data_y, aml_grid, bml_grid
    # )
    log_likelihood_mat = np.empty((len(aml_val), len(bml_val)))
    for i in range(len(aml_val)):
        for j in range(len(bml_val)):
            log_likelihood_mat[i, j] = linear_log_likelihood(
                data_x, data_y, aml_grid[i, j], bml_grid[i, j]
            )

    # Generate chi-sqaured value matrix
    # chi_sqr_mat = linear_chi_sqr(data_x, data_y, aml_grid, bml_grid)
    chi_sqr_mat = np.empty((len(aml_val), len(bml_val)))
    for i in range(len(aml_val)):
        for j in range(len(bml_val)):
            chi_sqr_mat[i, j] = linear_chi_sqr(
                data_x, data_y, aml_grid[i, j], bml_grid[i, j]
            )

    # Generate delta chi-squared matrix
    delta_chi_sqr_mat = chi_sqr_mat - np.min(chi_sqr_mat)

    # Locate the minimum delta chi-sqaured value's coordinate
    index_min = np.unravel_index(
        delta_chi_sqr_mat.argmin(), delta_chi_sqr_mat.shape
    )
    # Locate the best parameter pair based on the minimum index
    alm_min, blm_min = aml_val[index_min], bml_val[index_min]
    # Transform delta chi-squared grid into array

    # Plot for credible regions
    return print(alm_min, blm_min)
