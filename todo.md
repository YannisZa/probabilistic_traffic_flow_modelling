# Bugs

## Urgent

- Make De Romph's thermodynamic integration mcmc to converge

## Non-urgent


# Extensions

## Urgent
- Implement the following Fundamental Diagrams:
 - [x]: Exponential FD
 - [x]: Greenshield's FD
 - [x]: Daganzo's FD
 - [x]: Del Castillo's FD 
 - [x]: Greenberg's FD
 - [x]: Underwood's FD
 - [x]: Northwestern's FD
 - [x]: Newell's FD
 - [x]: Wang's FD
 - [x]: Smulder's velocity FD
 - []: DeRomph's FD

- Ammend FD-specific experiment toml files to reflect all existing datasets and inference metadata

## Non-urgent
- Rename vanilla mcmc to metropolis_hastings
- Write validate attribute and parameter statements where necessary
- Write tests for important functions

## Nice-to-have
- Allow construction of non-diagonal covariance matrices
- Allow transition kernel not to be symmetric - incorporate it into acceptance probability

# Tidying up

## Urgent

## Non-urgent
- Add explainer doc for toml files
- Readme file for running code 
- List of all FDs with their references

# Testing

## Urgent

## Non urgent
- All experiments using identity parameter transformation
- MLE parameter initialisation in vanilla/thermodynamic mcmc

