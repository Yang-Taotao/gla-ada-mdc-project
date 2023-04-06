## Radio data analysis
This is a course project for *PHYS-5001* at *University of Glasgow*.
- Drafted: Apr 06, 2023
- Editted: Apr 06, 2023

## Data set
We employ the data obtained from:
1. MDC1: Mock data challenge dataset 1
2. MDC2: Mock data challenge dataset 2

## Purpose
We conduct basic level data analysis and curve fits:
1. Linear least squares fitting
2. Maximum likelihood fitting

## Steps
We will perform the tasks with the following steps:
1. Load MDC datasets
2. Generate linear least squares fit results for MDC1
3. Generate and plot maximum likelihood fit results for MDC1
4. Perform MCMC algorithm on MDC1 and generate corner plot
5. Generate and plot quadratic maximum likelihood fit for MDC2
6. Get marginal likelihood and Bayes factor for MDC2

## Working file structure
Main file:
- main.py

Data reader modules:
- data_reader.py

Data fitter modules:
- data_fitter.py

Legacy maaster file:
- main_legacy.py

## Supplementary files
Readme:
- README.md

## Folder structure
Data folder:
- ./data/

Plot folder:
- ./media/
