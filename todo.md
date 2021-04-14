# Bugs

## Urgent
- Check if Northwestern's needs an implicit constraint rho_j > max_x


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
 - []: Smulder's velocity FD
 - []: DeRomph's FD

- Implement discontinuous versions of Daganzo's, Smulder's, De Romph's ?

- Ammend FD-specific experiment toml files to reflect all existing datasets and inference metadata


## Non-urgent
- Write validate attribute and parameter statements where necessary
- Write tests for important functions

## Nice-to-have
- Allow construction of non-diagonal covariance matrices
- Allow transition kernel not to be symmetric - incroporate it into acceptance probability

# Tidying up

## Urgent

## Non-urgent

# Testing

## Urgent
- Exponential FD inference using updated code and toml files

## Non urgent
- All experiments using identity parameter transformation
- MLE parameter initialisation in vanilla/thermodynamic mcmc
