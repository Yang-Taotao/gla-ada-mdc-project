"""
This is the data fitter module file for MDC data reading.
Results print out formatted with f-string methods.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Library import
# Import numpy with np alias for array manipulation
import numpy as np

# %% Log likelihood function repo
# Log likelihood function generator
def log_ll(data_x, data_y, param, model="Linear"):
    """
    Parameters
    ----------
    data_x : array
        Data x array.
    data_y : array
        Data y array.
    param : tuple
        Fit model parameters tuple.
    model : string
        Model selector string, defaul at "Linear".

    Returns
    -------
    log likelihood function
        Callable log likelihood calculation.
    """
    # Model selection
    # Select linear model
    if model == "Linear":
        # Local varible repo
        param_a, param_b = param
        # Generate linear model
        fit_model = param_a + param_b * data_x
    # Select quadratic model
    elif model == "Quadratic":
        # Local varible repo
        param_a, param_b, param_c = param
        # Generate quadratic model
        fit_model = param_a + param_b * data_x + param_c * data_x**2

    # Get stdev of residuals
    fit_sigma = np.std(data_y - fit_model)
    
    # Return callable linear log likelihood calculation result
    return -0.5 * np.sum(
        ((data_y - fit_model) ** 2) / (2 * fit_sigma**2)
        + np.log(2 * np.pi * fit_sigma**2)
    )


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
    print(f"{'Linear-ml fit result:':<30}")
    print("=" * 30)
    print(f"{'Fitted intercept:':<20}{fit_param[0]:>10.4g}")
    print(f"{'Fitted slope:':<20}{fit_param[1]:>10.4g}")
    print(f"{'Min of chi2:':<20}{min_chi2:>10.4g}")
    print("=" * 30)
    print()

    # Return function call result
    return result


# %% Metropolis - MCMC
# Linear MCMC function
def mcmc_fitter(data_x, data_y, model="Linear"):
    """
    Parameters
    ----------
    data_x : array
        Data array of x.
    data_y : array
        Data array of y.
    fit_ref : tuple
        Fit reference tuple from ls fitter.
    model : string
        Model selector, default at "Linear".

    Returns
    -------
    data_mcmc : array
        MCMC a, b pair array for corner plottings.
    """
    # Local variable repo, define total iterations and init counter
    data_step, data_accept = 100000, 0

    # Model selection
    # Select linear model
    if model == "Linear":
        # Generate empty mcmc sample, 2 columns, all values are zero    
        data_mcmc = np.zeros((data_step, 2))
        # Initialize chains with initial values
        data_mcmc[0], fit_sigma = (
            # First a, b value guess from ls fitting results
            [8.3, 2.6],
            # Sigma of a, b guesses import from ls fit results
            [0.01, 0.01],
        )
    # Select quadratic model
    elif model == "Quadratic":
        # Generate empty mcmc sample, 3 columns, all values are zero
        data_mcmc = np.zeros((data_step, 3))
        # Initialize chains with initial values
        data_mcmc[0], fit_sigma = (
            # First a, b value guess from ls fitting results
            [8.47, 2.55, 0.30],
            # Sigma of a, b guesses import from ls fit results
            [0.01, 0.01, 0.01],
        )

    # Generate empty log likelihood chains
    data_ll = np.zeros(data_step)
    # Initialize log likelihood and deposit results and a, b pairs
    data_ll[0] = log_ll(data_x, data_y, data_mcmc[0], model)

    # MCMC loop
    for i in range(1, data_step):
        
        # Model selection
        # For linear model
        if model == "Linear":
            # Get temp a, b value by shifting randomly from base a, b values
            temp = (
                data_mcmc[i - 1][0] + fit_sigma[0] * np.random.randn(1),
                data_mcmc[i - 1][1] + fit_sigma[1] * np.random.randn(1),
            )
        # For quadratic model
        elif model == "Quadratic":
            # Get temp a, b, c value by shifting randomly from base a, b, c values
            temp = (
                data_mcmc[i - 1][0] + fit_sigma[0] * np.random.randn(1),
                data_mcmc[i - 1][1] + fit_sigma[1] * np.random.randn(1),
                data_mcmc[i - 1][2] + fit_sigma[2] * np.random.randn(1),
            )

        # Get temp log likelihood from temp parameter assembly
        temp_ll = log_ll(data_x, data_y, temp, model)

        # Ratio analysis
        # Generate ratio from ll
        mcmc_ratio = np.exp(temp_ll - data_ll[i - 1])
        # Accept and continue
        if mcmc_ratio > np.random.rand(1) or mcmc_ratio >= 1:
            # Update base state with current temp state values
            data_mcmc[i], data_ll[i] = temp, temp_ll
            # Update counter
            data_accept += 1
        # Reject and revert to previous state
        else:
            # Update base state
            data_mcmc[i], data_ll[i] = (
                data_mcmc[i - 1],  # Revert to previous base parameter assembly
                data_ll[i - 1],  # Revert to previous base log likelihood
            )

    # Model selection
    # For linear model
    if model == "Linear":
        # Fit result tuple construction
        fit_result = (
            # Overall acceptance rate
            data_accept / data_step,
            # Fitted a, b value array average value
            (np.mean(data_mcmc[:, 0]), np.mean(data_mcmc[:, 1])),
            # Fitted stdev of a and b array
            (np.std(data_mcmc[:, 0]), np.std(data_mcmc[:, 1])),
            # Fitted cov of a and b array
            np.cov(data_mcmc[:, 0], data_mcmc[:, 1])[0, 1],
        )

        # Results printout
        print()
        print(f"{'Linear-MCMC result:':<30}")
        print("=" * 30)
        print(f"{'MCMC attempts:':<20}{data_step:>10.4g}")
        print(f"{'Acceptance rate:':<20}{fit_result[0]:>10.4g}")
        print(f"{'Mean intercept:':<20}{fit_result[1][0]:>10.4g}")
        print(f"{'Mean slope:':<20}{fit_result[1][1]:>10.4g}")
        print(f"{'Std intercept:':<20}{fit_result[2][0]:>10.4g}")
        print(f"{'Std slope:':<20}{fit_result[2][1]:>10.4g}")
        print(f"{'Covariance:':<20}{fit_result[3]:>10.4g}")
        print("=" * 30)
        print()

    # For quadratic model
    elif model == "Quadratic":
        # Fit result tuple construction
        fit_result = (
            # Overall acceptance rate
            data_accept / data_step,
            # Fitted a, b value array average value
            (np.mean(data_mcmc[:, 0]), np.mean(data_mcmc[:, 1]), np.mean(data_mcmc[:, 2])),
            # Fitted stdev of a and b array
            (np.std(data_mcmc[:, 0]), np.std(data_mcmc[:, 1]), np.std(data_mcmc[:, 2])),
        )

        # Results printout
        print()
        print(f"{'Quadratic-MCMC result:':<30}")
        print("=" * 30)
        print(f"{'MCMC attempts:':<20}{data_step:>10.4g}")
        print(f"{'Acceptance rate:':<20}{fit_result[0]:>10.4g}")
        print(f"{'Mean a:':<20}{fit_result[1][0]:>10.4g}")
        print(f"{'Mean b:':<20}{fit_result[1][1]:>10.4g}")
        print(f"{'Mean c:':<20}{fit_result[1][2]:>10.4g}")
        print(f"{'Std a:':<20}{fit_result[2][0]:>10.4g}")
        print(f"{'Std b:':<20}{fit_result[2][1]:>10.4g}")
        print(f"{'Std c:':<20}{fit_result[2][2]:>10.4g}")
        print("=" * 30)
        print()

    # Return results for plotter
    return data_mcmc
