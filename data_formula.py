"""
This is the data formula module file for MDC data reading.
Document the log likelihood functions.

Created on Mon Apr 10 2023

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
        Data x value array.
    data_y : array
        Data y value array.
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

    # Calculate log likelihood
    result = -0.5 * np.sum(
        ((data_y - fit_model) ** 2) / (2 * fit_sigma**2)
        + np.log(2 * np.pi * fit_sigma**2)
    )

    # Return log likelihood calculation result
    return result
