"""
This is the data fitter module file for MDC data reading.
Results print out formatted with f-string methods.
Focused on linear model fittings for 1.1 and 1.2.

Created on Mon Apr 10 2023

@author: Yang-Taotao
"""
# %% Library import
# Import numpy with np alias for array manipulation
import numpy as np

# Import custom log likelihood function module
from data_formula import log_ll

# %% Linear least squares fitter
# Linear ordinary least squares fitter
def linear_ls(data_x, data_y):
    """
    Parameters
    ----------
    data_x : array
        An array of x data.
    data_y : array
        An array of y data.

    Returns
    -------
    result : tuple
        A tuple of all fit results.
    """
    # Intermidiate summation calculation, results packed into tuple
    sum_res = (
        len(data_x),  # Sample size
        np.sum(data_x),  # Sum of x
        np.sum(data_y),  # Sum of y
        np.sum(data_x**2),  # Sum of the squared value of x
        np.sum(data_y**2),  # Sum of the squared value of y
        np.sum(data_x * data_y),  # Sum of x times y
    )
    # Intermidiate calculation of denominators, results packed into tuple
    denom_res = (
        # Denominator for least squares
        (sum_res[0] * sum_res[3] - (sum_res[1]) ** 2),
        # Denominator for corr of x
        np.sqrt(sum_res[0] * sum_res[3] - (sum_res[1]) ** 2),
        # Denominator for corr of y
        np.sqrt(sum_res[0] * sum_res[4] - (sum_res[2]) ** 2),
    )

    # Generate least squares fitting parameters a,b of linear model
    fit_param = (
        # Fit param a
        (sum_res[2] * sum_res[3] - sum_res[5] * sum_res[1]) / denom_res[0],
        # Fit param b
        (sum_res[0] * sum_res[5] - sum_res[2] * sum_res[1]) / denom_res[0],
    )

    # Calculate residual terms and its standard error
    # Get residual data array
    residual = data_y - (fit_param[0] + fit_param[1] * data_x)
    # Calculate stdev of fit result
    sigma = np.std(residual)

    # Calculate the variance of linear fit param a, b
    fit_var = (
        # Var of fit param a
        ((sigma**2) * sum_res[3]) / denom_res[0],
        # Var of fit param b
        ((sigma**2) * sum_res[0]) / denom_res[0],
    )
    # Calculate standard deviation of linear fit param a, b
    fit_sigma = (np.sqrt(fit_var[0]), np.sqrt(fit_var[1]))
    # Calculate covariance of linear fit
    fit_cov, fit_corr = (
        # Cov of fit param a, b
        ((-(sigma**2) * sum_res[1]) / denom_res[0]),
        # Corr of dataset
        (
            (sum_res[0] * sum_res[5] - sum_res[1] * sum_res[2])
            / (denom_res[1] * denom_res[2])
        ),
    )

    # Generate results, only fit parameters are of concerns
    result = fit_param, fit_sigma

    # Print results
    print()
    print(f"{'#1.1 - Linear-ls fit result:':<30}")
    print("=" * 30)
    print(f"{'Fitted intercept:':<20}{fit_param[0]:>10.4g}")
    print(f"{'Fitted slope:':<20}{fit_param[1]:>10.4g}")
    print(f"{'Var intercept:':<20}{fit_var[0]:>10.4g}")
    print(f"{'Var slope:':<20}{fit_var[1]:>10.4g}")
    print(f"{'Stdev intercept:':<20}{fit_sigma[0]:>10.4g}")
    print(f"{'Stdev slope:':<20}{fit_sigma[1]:>10.4g}")
    print(f"{'Covariance:':<20}{fit_cov:>10.4g}")
    print(f"{'Corr-coeff:':<20}{fit_corr:>10.4g}")
    print("=" * 30)
    print()

    # Return function call result
    return result


# %% Maximum likelihood fitter
# Linear maximum likelihood fitter
def linear_ml(data_x, data_y, fit_param):
    """
    Parameters
    ----------
    data_x : array
        An array of x data
    data_y : array
        An array of y data

    Returns
    -------
    result : tuple
        A tuple of all fit results
    """
    # Assign a, b value guesses, reference value from linear fit
    data_a, data_b = (
        # Value array of a guesses from linear fit
        np.linspace(fit_param[0] - 0.25, fit_param[0] + 0.25, 100),
        # Value array of b guesses from linear fit
        np.linspace(fit_param[1] - 0.25, fit_param[1] + 0.25, 100),
    )

    # Compute the log likelihood grid
    grid_ll = np.array(
        # Loop with list comprehension
        [
            # Cache log likelihood value of a, b pair
            log_ll(data_x, data_y, (data_a[i], data_b[j]), "Linear")
            # Loop through entries of a
            for i in range(len(data_a))
            # Loop through entries of b
            for j in range(len(data_b))
        ]
        # Reshape 1-D array into 2-D array with a, b dimensions
    ).reshape(len(data_a), len(data_b))

    # Compute the chi2 grid from log likelihood grid, chi2 = -2*ll
    grid_chi2 = (-2) * grid_ll

    # Locate the minimum chi2 result index with np.unravel
    idx_min = np.unravel_index(np.argmin(grid_chi2), grid_chi2.shape)
    # Index out the a, b pair at minimum chi2
    fit_param = data_a[idx_min[0]], data_b[idx_min[1]]
    # Get the minimum chi2 value
    min_chi2 = grid_chi2[idx_min]

    # Compute the delta chi2 grid into baysian percentage
    grid_delta_chi2 = grid_chi2 - min_chi2

    # Generate fit result tuple for plotting
    result = (data_a, data_b, grid_delta_chi2)

    # Results printout
    print()
    print(f"{'#1.2 - Linear-ml fit result:':<30}")
    print("=" * 30)
    print(f"{'Fitted intercept:':<20}{fit_param[0]:>10.4g}")
    print(f"{'Fitted slope:':<20}{fit_param[1]:>10.4g}")
    print(f"{'Min of chi2:':<20}{min_chi2:>10.4g}")
    print("=" * 30)
    print()

    # Return function call result
    return result
