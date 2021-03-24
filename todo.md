# Bugs

## Urgent
- Thermodynamic integration MCMC acceptance rate tuning

## Non-urgent
- Debug gelman and rubin criterion

# Extensions

## Urgent
- Implement sigma learning

## Non-urgent
- Allow transition kernel to sample from truncated gaussian
- Write validate attribute and parameter statements where necessary
- Change inference __init__.py to run parallel chains
- Write tests for important functions

## Nice-to-have
- Implement log posterior in C
- Allow transition kernel not to be symmetric - incroporate it into acceptance probability


# Tidying up

## Urgent

## Non-urgent
- Update test simulation and inference parameter toml files

# Testing functions
- Reject sample if theta is not wihtin lower and upper bounds
- MLE parameter initialisation in vanilla/thermodynamic mcmc