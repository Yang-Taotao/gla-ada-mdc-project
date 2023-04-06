"""
This is a script for MDC of ADA course.

Created on Fri Mar 10 2023

@author: Yang-Taotao
"""
# Initial data treatment
# Library import
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Load MDC part data parts and store as arrays
data_x1, data_y1 = np.loadtxt("MDC1.txt", unpack=True, usecols=(0, 1))
data_x2, data_y2 = np.loadtxt("MDC2.txt", unpack=True, usecols=(0, 1))

# Sample size cache
data_n1, data_n2 = len(data_x1), len(data_x2)

# Plot customization
plt.style.use(["science", "notebook", "grid"])

# MDC1.1
# Ordinary least squares fitting intermidiate calculations for sum
sum_x1, sum_y1, sum_x1_sqr, sum_y1_sqr, sum_x1y1 = (
    np.sum(data_x1),
    np.sum(data_y1),
    np.sum(data_x1**2),
    np.sum(data_y1**2),
    np.sum(data_x1 * data_y1),
)
# Ordinary least squares fitting intermidiate calculations for denominators
denom_ls, denom_corr_x1, denom_corr_y1 = (
    data_n1 * sum_x1_sqr - (sum_x1) ** 2,
    np.sqrt(data_n1 * sum_x1_sqr - (sum_x1) ** 2),
    np.sqrt(data_n1 * sum_y1_sqr - (sum_y1) ** 2),
)

# Ordinary least squares fitting parameter estimation
fit_als1, fit_bls1 = (
    (sum_y1 * sum_x1_sqr - sum_x1y1 * sum_x1) / denom_ls,
    (data_n1 * sum_x1y1 - sum_y1 * sum_x1) / denom_ls,
)
fit_param1 = fit_als1, fit_bls1

# Generate residual term and calculate its sigma
model_ls = fit_als1 + fit_bls1 * data_x1
fit_res1 = data_y1 - model_ls
data_sigma1 = np.std(fit_res1)
# Calculate the variance, standard deviation, and covariance of linear fit
fit_var_als1, fit_var_bls1, = (
    ((data_sigma1**2) * sum_x1_sqr) / denom_ls,
    ((data_sigma1**2) * data_n1) / denom_ls,
)
fit_sigma_als1, fit_sigma_bls1 = (np.sqrt(fit_var_als1), np.sqrt(fit_var_bls1))
fit_cov_als1bls1, fit_corr_x1y1 = (
    (-(data_sigma1**2) * sum_x1) / denom_ls,
    (data_n1 * sum_x1y1 - sum_x1 * sum_y1) / (denom_corr_x1 * denom_corr_y1),
)

# Print results for part 1.1
print(f"{'MDC1.1:':<40}")
print(f"{'Fitted intercept:':<40}{fit_als1:>15.4f}")
print(f"{'Fitted slope:':<40}{fit_bls1:>15.4f}")
print(f"{'Variance of intercept:':<40}{fit_var_als1:>15.4f}")
print(f"{'Variance of slope:':<40}{fit_var_bls1:>15.4f}")
print(f"{'Standard deviation of intercept:':<40}{fit_sigma_als1:>15.4f}")
print(f"{'Standard deviation of slope:':<40}{fit_sigma_bls1:>15.4f}")
print(f"{'Covariance of MDC1 fit:':<40}{fit_cov_als1bls1:>15.4f}")
print(f"{'Correlation coeff of MDC1 data:':<40}{fit_corr_x1y1:>15.4f}")

# Plot the fit
# plt.scatter(data_x1, data_y1, label="MDC1 data points")
plt.errorbar(
    data_x1,
    data_y1,
    yerr=abs(fit_res1),
    alpha=0.33,
    capsize=2,
    label="MDC1 data points",
)
plt.plot(
    data_x1,
    model_ls,
    color="red",
    label="MDC1 linear fitted line",
)
plt.title("Ordinary linear least squares fit result for MDC1")
plt.xlabel("MDC1 data $x$"), plt.ylabel("MDC1 data $y$"), plt.legend()
