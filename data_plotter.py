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

    # Get sigma levels from lecture slide 6
    levels = [2.30, 6.17, 11.8]

    # Plot generation
    fig, ax = plt.subplots()
    CS = ax.contour(data_b, data_a, delta_grid, levels=levels)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel("Value of parameter $b$")
    ax.set_ylabel("Value of parameter $a$")

    # Save and close
    fig.savefig("./media/fig_1_baysian.png")
    plt.show()
    plt.close()
