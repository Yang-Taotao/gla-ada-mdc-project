"""
This is the data plotter module file for MDC data reading.
List comprehension and tuple unpacking used for code organization.
The plt runs are closed at the end to prevent contaminations.

Created on Thu Apr 06 2023

@author: Yang-Taotao
"""
# %% Library import
# Import matplotlib with plt alias for plotting
import matplotlib.pyplot as plt

# Import corner package for MCMC corner plottin
import corner

# Import scienceplots package for plot customization
import scienceplots

# %%  Plot style config
# Plot style configuration to use Jupyter Notebook style plots with grids
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
        data_a,
        data_b,
        delta_grid,
        cmap=color,  # For contour color
        levels=levels,  # Add specific contour levels
    )
    # Contour plot for all delta chi2 grid levels with filled background
    plot_delta = plt.contourf(
        data_a,
        data_b,
        delta_grid,
        cmap=color,  # For contour fill color
        alpha=0.66,  # Set background color opacity
    )

    # Plot customization
    # Contour labeling at font 10 with inline displays
    plt.clabel(plot_level, levels, inline=True, fontsize=10)
    plt.clabel(plot_delta, fontsize=10)
    # Plot x,y axis labeling at font 14, render LaTeX with r
    plt.xlabel(r"Value of parameter $a$", fontsize=14)
    plt.ylabel(r"Value of parameter $b$", fontsize=14)
    # Colorbar
    plt.colorbar()

    # Save and close
    plt.savefig("./media/fig_1_baysian.png")
    plt.close()


# %% Corner plotter for linear MCMC
def mcmc_plotter(data_mcmc, model="Linear"):
    """
    Parameters
    ----------
    data_mcmc : array
        MCMC data array.
    model : string
        Model selection array, default at "Linear".

    Returns
    -------
    None.
    """
    # Model selection
    # For linear model
    if model == "Linear":
        # Set a, b axis label, render LaTeX, and set save path
        label, file_path = [
            r"Value of $a$",
            r"Value of $b$",
        ], "./media/fig_2_mcmc_linear.png"
    # For quadratic model
    elif model == "Quadratic":
        # Set a, b, c axis label, render LaTeX, and set save path
        label, file_path = [
            r"Value of $a$",
            r"Value of $b$",
            r"Value of $c$",
        ], "./media/fig_3_mcmc_quadratic.png"

    # Set 65% quantiles
    quantile = [0.16, 0.84]

    # Plot the corner plot with 65% quantiles and labels at size 10
    corner.corner(
        data_mcmc,
        labels=label,
        quantiles=quantile,
        max_n_ticks=5,  # Limit axis tick numbers
        show_titles=True,  # Add title display
        title_fmt=".4g",  # To change math display to 4 sig fig
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 10},
        plot_density=True,  # Force default option
        plot_datapoints=True,  # Force default option
    )

    # Plot customization, add ticks to x, y axis at font 14
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save and close
    plt.savefig(file_path)
    plt.close()
