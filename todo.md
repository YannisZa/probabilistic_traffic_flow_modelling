# Bugs

Parameter posterior mean bias

What is going wrong?
- MLE and Metropolis Hastings MCMC parameter posterior mean are biased

What can I check?
[x] Likelihood is correctly defined
[x] Data is correctly simulated
[x] Noise is not too large
[x] Tweaking proposal parameters 
[x] Number of data points is sufficient
[x] Random seed is set to None
[] Is likelihood not identifiable? 
[] Try transforming alpha to phi = exp(alpha)

Things to try out today
[x] Compute MLE Jacobian to plug into scipy minimize and get unbiased MLE [COULD NOT BE DONE - MLE of one parameter is written in terms of the other]
[x] Effect of Jacobian transformation on proposal in MCMC
[] Poisson estimator


## Urgent

## Non-urgent
- Debug gelman and rubin criterion

# Extensions

## Urgent
- Change inference __init__.py to run parallel chains
- Implement sigma learning

## Non-urgent
- Allow transition kernel to sample from truncated gaussian
- Write validate attribute and parameter statements where necessary
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