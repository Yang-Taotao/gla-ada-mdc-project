"""
This is the data plotter module file for MDC data reading.
List comprehension and tuple unpacking used for code organization.

Created on Thu Apr 06 2023

@author: Yang-Taotao
"""
# %% Library import
import numpy as np
import matplotlib.pyplot as plt
import corner
import scienceplots

# %%  Plot style config
# Plot style configuration
plt.style.use(["science", "notebook", "grid"])

# %% Baysian credible region plotter
def baysian_plotter(arg):
    """
    Parameters
    ----------
    arg : tuple
        Plotter argument tuple.

    Returns
    -------
    None.
    """
    # Local varible repo of contour plot dataset
    data_a, data_b, delta_grid = [arg[i] for i in range(len(arg))]

    # Define sigma levels from lecture slide 6 and the color map
    levels, color = ([2.30, 6.17, 11.8], "plasma")

    # Plot generation
    # Main contour plot with specified values
    plot_level = plt.contour(
        data_b, data_a, delta_grid, cmap=color, levels=levels
    )
    # Contour plot for all delta chi2 grid levels with filled background
    plot_delta = plt.contourf(
        data_b, data_a, delta_grid, cmap=color, alpha=0.66
    )

    # Plot customization
    # Contour labeling
    plt.clabel(plot_level, levels, inline=True, fontsize=10)
    plt.clabel(plot_delta, fontsize=10)
    # Plot x,y axis labeling
    plt.xlabel("Value of parameter $b$", fontsize=14)
    plt.ylabel("Value of parameter $a$", fontsize=14)
    # Colorbar
    plt.colorbar()

    # Save and close
    plt.savefig("./media/fig_1_baysian.png")
    plt.close()


# %% Corner plotter for linear MCMC
def linear_mcmc_plotter(data_mcmc):

    # Plot param repo
    label, quantile = (
        # Set a, b axis label, render LaTeX
        [r"Value of $a$", r"Value of $b$"],
        # Set 65% quantiles
        [0.16, 0.84],
    )

    # Plot the corner plot
    corner.corner(
        data_mcmc,
        labels=label,
        quantiles=quantile, 
        max_n_ticks=5,
        show_titles=True,
        title_fmt=".4g",
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 10},
        plot_density=True,
        plot_datapoints=True,
    )

    # Plot customization, add ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save and close
    plt.savefig("./media/fig_2_mcmc.png")
    plt.close()
