"""
This is the data fitter module file for MDC data reading.
Results print out formatted with f-string methods.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Library import
import numpy as np

# %% Linear least squares fitter
# Linear ordinary least squares fitter
def linear_ls(data_x, data_y):
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
        (
            sum_res[0] * sum_res[3] - (sum_res[1]) ** 2
        ),
        # Denominator for corr of x
        np.sqrt(
            sum_res[0] * sum_res[3] - (sum_res[1]) ** 2
        ),
        # Denominator for corr of y
        np.sqrt(
            sum_res[0] * sum_res[4] - (sum_res[2]) ** 2
        ),
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
        (
            (-(sigma**2) * sum_res[1]) / denom_res[0]
        ),
        # Corr of dataset
        (
            (sum_res[0] * sum_res[5] - sum_res[1] * sum_res[2])
            / (denom_res[1] * denom_res[2])
        ),
    )

    # Generate results tuple
    result = (fit_param, fit_sigma, fit_var, fit_cov, fit_corr)

    # Print results
    print()
    print(f"{'Linear-ls fit result:':<30}")
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
def linear_ml(data_x, data_y):
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
    # Linear log likelihood function generator
    def linear_ll(param):
        # Unpack linear model param
        param_a, param_b = param
        # Generate linear model y = a+b*x
        fit_model = param_a + param_b * data_x
        # Get stdev of residuals
        fit_sigma = np.std(data_y - fit_model)
        # Return callable linear log likelihood calculation result
        return (
            -0.5 * np.sum((data_y - fit_model)**2 / fit_sigma**2 + np.log(2*np.pi*fit_sigma**2))
        )

    # Assign a, b value guesses
    a_val, b_val = (
        # Value array of a
        np.linspace(np.min(data_y), np.max(data_y), 100),
        # Value array of b
        np.linspace(-1, 1, 100),
    )

    # Compute the log likelihood grid
    grid_ll = np.array(
        [
            linear_ll([a_val[i], b_val[j]]) 
            for i in range(len(a_val)) 
            for j in range(len(b_val))
        ]
    ).reshape(len(a_val), len(b_val))

    # Compute the chi2 grid
    grid_chi2 = -2 * grid_ll
    
    # Locate the minimum chi2 result
    idx_min = np.unravel_index(np.argmin(grid_chi2), grid_chi2.shape)
    # Index out the a, b pair at minimum chi2
    fit_param = a_val[idx_min[0]], b_val[idx_min[1]]
    # Get the minimum chi2 value
    min_chi2 = grid_chi2[idx_min]
    
    # Compute the delta chi2 grid
    grid_delta_chi2 = grid_chi2 - min_chi2
    # Get the minimum value of delta chi2
    min_delta_chi2 = np.argmin(grid_delta_chi2)

    # Generate fit result tuple for plotting
    result = (fit_param[0], fit_param[1], grid_delta_chi2)

    # Results printout
    print()
    print(f"{'Linear-ml fit result:':<30}")
    print("=" * 30)
    print(f"{'Fitted intercept:':<20}{fit_param[0]:>10.4g}")
    print(f"{'Fitted slope:':<20}{fit_param[1]:>10.4g}")
    print(f"{'Min of chi2:':<20}{min_chi2:>10.4g}")
    print(f"{'Min of delta chi2:':<20}{min_delta_chi2:>10.4g}")
    print("=" * 30)
    print()

    # Return function call result
    return result

