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
    param_a, param_b, delta_grid = [arg[i] for i in range(len(arg))]

    # Get sigma levels from lecture slide 6
    levels = [2.30, 6.17, 11.8]

    # Plot generation
    plt.contour(param_b, param_a, delta_grid, levels=levels)
    plt.xlabel('Value of parameter $b$')
    plt.ylabel('Value of parameter $a$')
    plt.savefig("./media/fig_1_baysian.png")