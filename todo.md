# Bugs

## Urgent
- Log posterior grid search

## Non-urgent

# Extensions

## Urgent
- Move Show titles to experiments
- Change inference __init__.py to run parallel chains
- Implement sigma learning

## Non-urgent
- Allow transition kernel to sample from truncated gaussian
- Write validate attribute and parameter statements where necessary
- Write tests for important functions

## Nice-to-have
- Function that ensures no wrong inputs from toml files
- Allow onstruction of non-diagonal covariance matrices
- Allow transition kernel not to be symmetric - incroporate it into acceptance probability

# Tidying up

## Urgent

## Non-urgent
- Update test simulation and inference parameter toml files

# Testing functions
- Reject sample if theta is not wihtin lower and upper bounds
- MLE parameter initialisation in vanilla/thermodynamic mcmc