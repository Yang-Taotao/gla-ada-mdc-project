"""
This is the module file for performaing linear ordinary least squares fitting.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# Library import
import numpy as np

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
