# Transition kernel for vanilla MCMC
print('p',[0.0,0.0])
transition_kernel_samples = np.array([inf_model.propose_new_sample([0.0,0.0]) for n in range(10000)])
print(transition_kernel_samples.shape)

print('alpha')
print('mean',np.mean(transition_kernel_samples[:,0]))
print('std',np.std(transition_kernel_samples[:,0]))
print('beta')
print('mean',np.mean(transition_kernel_samples[:,1]))
print('std',np.std(transition_kernel_samples[:,1]))

plt.figure(figsize=(10,10))
plt.title('alpha')
plt.hist(transition_kernel_samples[:,0],bins=50)
plt.show()
plt.figure(figsize=(10,10))
plt.title('beta')
plt.hist(transition_kernel_samples[:,1],bins=50)
plt.show()

# Vanilla MCMC routine

lp_prev = self.evaluate_log_joint_prior(p_prev)
ll_prev = self.evaluate_log_likelihood(p_prev)

lp_new = self.evaluate_log_joint_prior(p_new)
ll_new = self.evaluate_log_likelihood(p_new)/


print('p_prev',p_prev,'lf_prev',lt_prev)
print('p_new',p_new,'lf_new',lt_new)
print('lp_prev',lp_prev)
print('lp_new',lp_new)
print('ll_prev',ll_prev)
print('ll_new',ll_new)
print(f'Acceptance rate {int(100*acc / N)}%')
sys.exit(1)

###################################################################################################
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import numpy as np
import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

inference_id = "grwmh_inference_wide_gamma_priors_sigma2_fixed_smaller_more_data"

# Instantiate objects
inf_model = utils.instantiate_inference_method(inference_id)
fd = utils.instantiate_fundamental_diagram(inf_model.inference_metadata['data_id'])

print(inference_id)
print(inf_model.inference_metadata['data_id'])

# Populate them with data
fd.populate()
inf_model.populate(fd)

# Fix random seed
np.random.seed(2021)

N = 20000
beta_step = 0.01
K = np.diag([0.00225,0.0000225])
p0 = [0.4,0.2]
sigma2 = fd.true_parameters[2]
# Initialise output variables
theta = []
theta_proposed = []
acc = 0
# Store necessary parameters
p_prev = p0
# Flag for printing
prints = True
print('p0',p_prev)
print(f'Running MCMC with {N} iterations')

def log_prior(p):
    return ss.gamma.logpdf(p[0],6.0,scale=0.1)+ ss.gamma.logpdf(p[1],2,scale=0.05)
def log_likelihood(p,q):
     return (ss.multivariate_normal.logpdf(q,mean=np.log(fd.simulate(p)),cov=sigma2*np.eye(len(q))) - np.sum(np.log(q)))
def log_posterior(p,q):
    return log_prior(p) + log_likelihood(p,q)

# Loop through MCMC iterations
for i in tqdm(range(N)):

    # Evaluate log function for current sample
    lt_prev = log_posterior(p_prev,fd.q)

    # Propose new sample
    p_new = p_prev + beta_step * np.random.multivariate_normal(np.zeros(2),K)

    # Evaluate log function for proposed sample
    lt_new = log_posterior(p_new,fd.q)

    # Printing proposals every 0.1*Nth iteration
    if prints and (i in [int(j/10*N) for j in range(1,11)]):
        print('p_prev',p_prev,'lf_prev',lt_prev)
        print('p_new',p_new,'lf_new',lt_new)

    # Calculate acceptance probability
    log_acc = lt_new - lt_prev
    # Sample from Uniform(0,1)
    log_u = np.log(np.random.random())

    # Accept/Reject
    # Compare log_alpha and log_u to accept/reject sample
    if min(np.exp(log_acc),1) >= np.exp(log_u):
        if prints and (i in [int(j/10*N) for j in range(1,11)]):
            print('p_new =',p_new)
            print('Accepted!')
            print(f'Acceptance rate {int(100*acc / N)}%')
        # Increment accepted sample count
        acc += 1
        # Append to accepted and proposed sample arrays
        theta.append(p_new)
        theta_proposed.append(p_new)
        # Update last accepted sample
        p_prev = p_new
    else:
        if prints and (i in [int(j/10*N) for j in range(1,11)]):
            print('Rejected...')
            print(f'Acceptance rate {int(100*acc / N)}%')
        # Append to accepted and proposed sample arrays
        theta.append(p_prev)
        theta_proposed.append(p_new)

# Update class attributes
theta = np.array(theta)
theta_proposed = np.array(theta_proposed)
print(f'Acceptance rate {int(100*acc / N)}%')

utils.ensure_dir(f"../data/output/debugging/{inf_model.inference_metadata['data_id']}/{inference_id}/")
np.savetxt(f"../data/output/debugging/{inf_model.inference_metadata['data_id']}/{inference_id}/theta.txt",theta)
np.savetxt(f"../data/output/debugging/{inf_model.inference_metadata['data_id']}/{inference_id}/theta_proposed.txt",theta_proposed)
