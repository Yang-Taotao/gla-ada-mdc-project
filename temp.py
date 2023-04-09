## Example MCMC script with corner plot
## Note that more comments are required when submitting for assessment.

## Necessary Imports
import numpy as np
import numpy.random as nprd
import matplotlib.pyplot as plt
import corner

from data_reader import file_loader

## Specify x and y data
# x = [1,2,3,4,5]
# y = [3.83,8.75,10.98,14.18,17.22]

x, y = file_loader("./data/MDC1.txt")

## Define sigma
sigma = np.std(y)

## Specify length of MCMC chain
nmcmc = 100000

## Empty entry init
a = np.zeros(nmcmc)
b = np.zeros(nmcmc)
samples = np.zeros((nmcmc,2))
chisq = np.zeros(nmcmc)

## Initial guesses repo
accept = 0

a[0] = 1.0
samples[0,0] = 1.0
sig_a = 0.1

b[0] = 1.0
samples[0,1] = 1.0
sig_b = 0.1

## Generate chi2 array from all a, b pairs
for k in range(0,len(x)):
    chisq[0] = chisq[0] + ((y[k]-(a[0]+b[0]*x[k]))/sigma)**2


## MCMC
for i in range(1,nmcmc):
    ## Progress bar
    print("Point "+str(i+1)+" of "+str(nmcmc), end="\r")

    ## Working a, b values
    a_trial = a[i-1] + sig_a*nprd.randn(1)
    b_trial = b[i-1] + sig_b*nprd.randn(1)
    ## Initial chi2 value of working a, b pair
    chisq_trial = 0

    ## Generate chi2 array of all working a, b pairs
    for k in range(0,len(x)):
        chisq_trial = chisq_trial + ((y[k]-(a_trial+b_trial*x[k]))/sigma)**2

    ## Compute the ratio from chi2 value --> Log Likelihood
    log_Lratio = 0.5*(chisq[i-1]-chisq_trial)

    ## Ratio analysis
    ## If accepted
    if log_Lratio >= 0:
        ## Update a, b values
        a[i] = a_trial
        b[i] = b_trial
        ## Cache a, b pair to sample
        samples[i,0] = a[i]
        samples[i,1] = b[i]
        ## Write in the current chi2 value
        chisq[i] = chisq_trial
        ## Update accept counter
        accept = accept + 1

    ## In case the ratio is not satisfactory
    else:
        ## Calculate the log likelihood ratio from chi2 ratio
        ratio = np.exp(log_Lratio)
        ## Generate uniform tester value
        test_uniform = nprd.rand(1)

        ## Conditional accept
        if test_uniform < ratio:
            ## Update a, b values
            a[i] = a_trial
            b[i] = b_trial
            ## Write to samples
            samples[i,0] = a[i]
            samples[i,1] = b[i]
            ## Write current chi2
            chisq[i] = chisq_trial
            ## Update accept counter
            accept = accept + 1

        ## Conditional reject
        else:
            ## Revert a, b to previous value pair
            a[i] = a[i-1]
            b[i] = b[i-1]
            ## Write reverted a, b pair back into sample
            samples[i,0] = a[i]
            samples[i,1] = b[i]
            ## Write current chi2
            chisq[i] = chisq[i-1]

## Calculate the overall acceptance ratio
accept_ratio = accept/nmcmc

## Plotting and information
## Do corner plot
figure = corner.corner(samples)
plt.show()
plt.savefig("./media/fig_2-mcmc_original.png")

## Caculate mean and stdev of all a, b guesses
bar = [np.mean(a), np.mean(b)]
err = [np.std(a), np.std(b)]

## Get covariance of a, b
cvr = np.cov(a,b)
covari = cvr[0,1]

## Print results
print("")
print("From MCMC")
print("a           b")
print(bar)
print(err)
print("Covariance")
print(covari)
print("Acceptance Ratio")
print(accept_ratio)