"""
This is the data fitter module file for MDC data reading.
Results print out formatted with f-string methods.
Focusing on the computation of Bayes factor.

Created on Mon Apr 10 2023

@author: Yang-Taotao
"""
# %% Library import
# Import numpy with np alias for array manipulation
import numpy as np

# Import custom log likelihood function module
from data_formula import log_ll

# Impot scipy for maximum summation
from scipy.special import logsumexp

# %% Gridded log likelihood grid function and Bayes factor calculation
def bayes_factor(data_x1, data_y1, data_x2, data_y2):
    """
    Parameters
    ----------
    data_x1 : array
        Data array of x value from MDC1.
    data_y1 : array
        Data array of y value from MDC1.
    data_x2 : array
        Data array of x value from MDC2.
    data_y2 : array
        Data array of y value from MDC2.

    Returns
    -------
    result : array
        Results return array.
    """
    # Manual generation of local a, b, c guesses from mcmc fitter
    fit_param = (
        # Value array of a guesses
        np.linspace(8.4 - 0.25, 8.4 + 0.25, 100),
        # Value array of b guesses
        np.linspace(2.5 - 0.25, 2.5 + 0.25, 100),
        # Value array of c guesses
        np.linspace(0.3 - 0.25, 0.3 + 0.25, 100),
    )

    # Generate log likelihood grid for linear and quadratic model
    grid_ll_line, grid_ll_quad = (
        # Generate linear log likelihood grid
        np.array(
            # Loop with list comprehension
            [
                # Cache log likelihood value of a, b pair
                log_ll(
                    data_x1,
                    data_y1,
                    (fit_param[0][i], fit_param[1][j]),
                    "Linear",
                )
                # Loop through entries of a
                for i in range(len(fit_param[0]))
                # Loop through entries of b
                for j in range(len(fit_param[1]))
            ]
            # Reshape 1-D array into 2-D array with a, b dimensions
        ).reshape(len(fit_param[0]), len(fit_param[1])),
        # Generate quadratic log likelihood grid
        np.array(
            # Loop with list comprehension
            [
                # Cache log likelihood value of a, b, c assembly
                log_ll(
                    data_x2,
                    data_y2,
                    (fit_param[0][i], fit_param[1][j], fit_param[2][k]),
                    "Quadratic",
                )
                # Loop through entries of a
                for i in range(len(fit_param[0]))
                # Loop through entries of b
                for j in range(len(fit_param[1]))
                # Loop through entries of c
                for k in range(len(fit_param[2]))
            ]
            # Reshape 1-D array into 3-D array with a, b, c dimensions
        ).reshape(len(fit_param[0]), len(fit_param[1]), len(fit_param[2])),
    )

    # Calculate the marginal log likelihood
    margi_ll = (
        # Generate marginal log likelihood sum for linear model grid
        # Normalize with number of linear log likelihood grid entries
        logsumexp(grid_ll_line) / (grid_ll_line.size),
        # Generate marginal log likelihood sum for quadratic model grid
        # Normalize with number of quadratic log likelihood grid entries
        logsumexp(grid_ll_quad) / (grid_ll_quad.size),
    )

    # Calculate the final Bayes factor
    bayes = np.exp(margi_ll[0] - margi_ll[1])

    # Bayes factor interpreter
    if bayes > 1:
        res_bayes = "Quadratic"
    elif bayes < 1:
        res_bayes = "Linear"
    else:
        res_bayes = "N/A"

    # Result printout
    print()
    print(f"{'#2.2 - Bayes factor result:':<30}")
    print("=" * 30)
    print(f"{'Line marginal ll:':<20}{margi_ll[0]:>10.4g}")
    print(f"{'Quad marginal ll:':<20}{margi_ll[1]:>10.4g}")
    print(f"{'Bayes factor:':<20}{bayes:>10.4g}")
    print(f"{'The better model:':<20}{res_bayes:>10}")
    print("=" * 30)
    print()

    # Result tuple generation
    result = bayes_factor

    # Return function call
    return result
