"""
This is the data fitter module file for MDC data reading.
Results print out formatted with f-string methods.

Created on Mon Mar 13 2023

@author: Yang-Taotao
"""
# %% Library import
# Import numpy with np alias for array manipulation
import numpy as np

# Import custom log likelihood function module
from data_formula import log_ll

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
    result : tuple
        MCMC result array for corner plottings.
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
            [np.mean(data_mcmc[:, 0]), np.mean(data_mcmc[:, 1])],
            # Fitted stdev of a and b array
            [np.std(data_mcmc[:, 0]), np.std(data_mcmc[:, 1])],
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
            # Fitted a, b, and c value array average value
            [
                np.mean(data_mcmc[:, 0]),
                np.mean(data_mcmc[:, 1]),
                np.mean(data_mcmc[:, 2]),
            ],
            # Fitted stdev of a, b, and c array
            [
                np.std(data_mcmc[:, 0]),
                np.std(data_mcmc[:, 1]),
                np.std(data_mcmc[:, 2]),
            ],
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

    # Construct return result of MCMC data array
    result = data_mcmc

    # Return results for plotter
    return result
