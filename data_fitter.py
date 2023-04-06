"""
This is the data fitter module file for MDC data reading.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
"""
This is the module file for performaing linear ordinary least squares fitting.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Library import
# Library import
import numpy as np
import matplotlib.pyplot as plt

# %% Least squares fitter
# Ordinary least squares fitting function
def linear_ls(data_x, data_y):
    """
    Parameters
    ----------
    data_x : numpy array
        An array of x data
    data_y : numpy array
        An array of y data

    Returns
    -------
    result : tuple
        A tuple of all fit results
    """
    # Intermidiate calculation of linear fitting
    data_n, sum_x, sum_y, sum_x_sqr, sum_y_sqr, sum_xy = (
        len(data_x),
        np.sum(data_x),
        np.sum(data_y),
        np.sum(data_x**2),
        np.sum(data_y**2),
        np.sum(data_x * data_y),
    )
    # Intermidiate calculation of denominators for linear fitting
    denom_ls, denom_corr_x, denom_corr_y = (
        data_n * sum_x_sqr - (sum_x) ** 2,
        np.sqrt(data_n * sum_x_sqr - (sum_x) ** 2),
        np.sqrt(data_n * sum_y_sqr - (sum_y) ** 2),
    )

    # Generate least squares fitting parameters of linear model
    als, bls = (
        (sum_y * sum_x_sqr - sum_xy * sum_x) / denom_ls,
        (data_n * sum_xy - sum_y * sum_x) / denom_ls,
    )
    fit_param = als, bls

    # Cakculate residual terms and its standard error
    model_ls = als + bls * data_x
    fit_res = data_y - model_ls
    sigma = np.std(fit_res)

    # Calculate the variance, standard deviation, and covariance of linear fit
    var_als, var_bls, = (
        ((sigma**2) * sum_x_sqr) / denom_ls,
        ((sigma**2) * data_n) / denom_ls,
    )
    sigma_als, sigma_bls = (np.sqrt(var_als), np.sqrt(var_bls))
    cov_alsbls, corr_xy = (
        (-(sigma**2) * sum_x) / denom_ls,
        (data_n * sum_xy - sum_x * sum_y) / (denom_corr_x * denom_corr_y),
    )

    # Generate results tuple
    result = (
        als,
        bls,
        fit_param,
        fit_res,
        sigma,
        var_als,
        var_bls,
        sigma_als,
        sigma_bls,
        cov_alsbls,
        corr_xy,
    )

    # Print results
    print()
    print(f"{'Linear Least Squared Fitting Result:':<30}")
    print("="*30)
    print(f"{'Fitted intercept:':<20}{als:>10.4g}")
    print(f"{'Fitted slope:':<20}{bls:>10.4g}")
    print(f"{'Var intercept:':<20}{var_als:>10.4g}")
    print(f"{'Var slope:':<20}{var_bls:>10.4g}")
    print(f"{'Stdev intercept:':<20}{sigma_als:>10.4g}")
    print(f"{'Stdev slope:':<20}{sigma_bls:>10.4g}")
    print(f"{'Covariance:':<20}{cov_alsbls:>10.4g}")
    print(f"{'Corr-coeff:':<20}{corr_xy:>10.4g}")
    print("="*30)
    print()
    return result

# %% Maximum likelihood fitter
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
