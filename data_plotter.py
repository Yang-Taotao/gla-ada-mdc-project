"""
This is the data plotter module file for MDC data reading.

Created on Thu Apr 06 2023

@author: Yang-Taotao
"""
# %% Library import
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# %%  Plot style config
# Plot style configuration
plt.style.use(["science", "notebook", "grid"])

# %% Baysian credible region plotter
def baysian_plotter(arg):
    # Local varible repo
    data_a, data_b, delta_grid = [arg[i] for i in range(len(arg))]

    # Define sigma levels from lecture slide 6 and the color map
    levels, color = ([2.30, 6.17, 11.8], "plasma")

    # Plot generation
    # Main plot with specified values
    plot_level = plt.contour(data_b, data_a, delta_grid, cmap=color, levels=levels)
    # Background plot for all delta chi2 grid levels
    plot_delta = plt.contourf(data_b, data_a, delta_grid, cmap=color, alpha=0.66)

    # Plot customization
    # Contour labeling 
    plt.clabel(plot_level, levels, inline=1, fontsize=10)
    plt.clabel(plot_delta, inline=1, fontsize=10)
    # Plot x,y axis labeling
    plt.xlabel("Value of parameter $b$", fontsize=14)
    plt.ylabel("Value of parameter $a$", fontsize=14)
    # Colorbar
    plt.colorbar()

    # Save and close
    plt.savefig("./media/fig_1_baysian.png")
    plt.close()
