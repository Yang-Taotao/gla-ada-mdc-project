"""
This is the data fitter module file for MDC data reading.
Results print out formatted with f-string methods.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Library import
import numpy as np
import matplotlib.pyplot as plt
import corner

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
    result = fit_param

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


# %% Log likelihood function repo
# Linear log likelihood function generator
def linear_ll(data_x, data_y, param_a, param_b):
    """
    Parameters
    ----------
    param_a : float
        Linear fit model parameter, intercept.
    param_b : float
        Linear fit model parameter, slope.

    Returns
    -------
    log likelihood function
        Callable log likelihood calculation.
    """
    # Generate linear model y = a+b*x
    fit_model = param_a + param_b * data_x
    # Get stdev of residuals
    fit_sigma = np.std(data_y - fit_model)
    # Return callable linear log likelihood calculation result
    return -0.5 * np.sum(
        (data_y - fit_model) ** 2 / fit_sigma**2
        + np.log(2 * np.pi * fit_sigma**2)
    )


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
            linear_ll(data_x, data_y, data_a[i], data_b[j])
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
def linear_mcmc(data_x, data_y):
    # Local variable repo
    data_step, fit_init, fit_step = ( 
        100000,
        1.0,
        0.1,
    )

    # Generate empty chains
    mcmc_a, mcmc_b, mcmc_ll = (
        [fit_init],
        [fit_init],
        [linear_ll(data_x, data_y, fit_init, fit_init)],
    )

    # MCMC
    for i in range(data_step):
        # Generate a new proposal for (a, b) using a Gaussian distribution
        temp_a = np.random.normal(mcmc_a[-1], fit_step)
        temp_b = np.random.normal(mcmc_b[-1], fit_step)

        # Calculate the log-likelihood of the proposed parameters
        temp_ll = linear_ll(temp_a, temp_b, data_x, data_y)

        # Calculate the acceptance ratio
        fit_ratio = temp_ll / mcmc_ll[-1]

        # Accept or reject the proposal
        if fit_ratio > np.random.rand():
            mcmc_a.append(temp_a)
            mcmc_b.append(temp_b)
            mcmc_ll.append(temp_ll)
        else:
            mcmc_a.append(mcmc_a[-1])
            mcmc_b.append(mcmc_b[-1])
            mcmc_ll.append(mcmc_ll[-1])

    # Discard the burn-in period (first half of the chain)
    mcmc_a = mcmc_a[data_step//2:]
    mcmc_b = mcmc_b[data_step//2:]

    # Compute the mean and standard deviation of the parameters
    a_mean = np.mean(mcmc_a)
    b_mean = np.mean(mcmc_b)
    a_std = np.std(mcmc_a)
    b_std = np.std(mcmc_b)

    # Plot the corner plot
    samples = np.vstack((mcmc_a, mcmc_b)).T
    labels = ['a', 'b']
    truths = [None, None]
    fig = corner.corner(samples, labels=labels, truths=truths)
    plt.show()

    return fig, a_mean, b_mean, a_std, b_std    