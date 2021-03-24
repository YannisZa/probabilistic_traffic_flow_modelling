# Bugs

## Urgent
- Thermodynamic integration MCMC
- Thermodynamic integration MCMC ML estimation

## Non-urgent
- Debug gelman and rubin criterion

# Extensions

## Urgent
- Assert data id in simulation metadata = data id in inference metadata
- Implement sigma learning

## Non-urgent
- Allow transition kernel to sample from truncated gaussian
- Write validate attribute and parameter statements where necessary
- Change inference __init__.py to run parallel chains
- Write tests for important functions

## Nice-to-have
- Function that updates a class attribute that does not exist
- Implement log posterior in C
- Allow transition kernel not to be symmetric - incroporate it into acceptance probability


# Tidying up

## Urgent

## Non-urgent
- Update test simulation and inference parameter toml files

# Testing functions
- Reject sample if theta is not wihtin lower and upper bounds
- MLE parameter initialisation in vanilla/thermodynamic mcmc