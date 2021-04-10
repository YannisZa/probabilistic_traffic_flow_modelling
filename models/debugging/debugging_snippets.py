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


##########################################################################################################################################################################
def generate_log_unnormalised_posteriors_plots(self,show_plot:bool=False,show_title:bool=True):

    # Get starting time
    start = time.time()

    # Make sure you have stored the necessary attributes
    utils.validate_attribute_existence(self,['log_unnormalised_posterior'])

    # Get number of plots
    num_plots = int(comb(len(self.log_unnormalised_posterior.shape),2))

    # Get plot combinations
    parameter_indices = list(itertools.combinations(range(0,self.num_learning_parameters), 2))

    # Transform paramter names
    transformed_parameters = []
    for i in range(self.num_learning_parameters):
        # Define parameter transformation
        transformation = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[i],latex_characters)]['transformation']
        # Append transformed parameter name to list
        transformed_parameters.append((f'${transformation}$'+self.parameter_names[i]))
    # Get parameter name combinations for each plot
    transformed_parameter_names = list(itertools.combinations(transformed_parameters, 2))

    # Avoid plotting more than 3 plots
    if num_plots > 3:
        raise ValueError(f'Too many ({num_plots}) log posterior plots to handle!')
    elif num_plots <= 0:
        raise ValueError(f'You cannot plot {num_plots} plots!')

    # print('Generating log posterior plots')
    # Loop through each plot
    figs = []

    for i in range(num_plots):
        index = parameter_indices[i]

        # Set Q_hat to log posterior
        Q_hat = self.log_unnormalised_posterior
        # Sum up dimension not plotted if there log posterior is > 2dimensional
        if len(Q_hat.shape) > 2:
            Q_hat = np.sum(Q_hat,axis=list(set(range(self.num_learning_parameters)) - set(index))[0])

        # Try to load plot parameters
        levels = None
        # Check if all plot parameters are not empty
        if all(bool(x) for x in self.inference_metadata['plot']['true_posterior'].values()):
            # Get number of colors in contour
            num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),2])
            # Update levels
            levels = np.linspace(float(self.inference_metadata['plot']['true_posterior']['vmin']),float(self.inference_metadata['plot']['true_posterior']['vmax']),num_colors)
        else:
            vmin = np.min(Q_hat)
            if bool(self.inference_metadata['plot']['true_posterior']['vmin']):
                vmin = np.max([float(self.inference_metadata['plot']['true_posterior']['vmin']),np.min(Q_hat)])
            vmax = np.max(Q_hat)
            if bool(self.inference_metadata['plot']['true_posterior']['vmax']):
                vmax = np.min([float(self.inference_metadata['plot']['true_posterior']['vmax']),np.max(Q_hat)])
            num_colors = int(np.sqrt(np.prod(Q_hat.shape)))
            if bool(self.inference_metadata['plot']['true_posterior']['num_colors']):
                # Get number of colors in contour
                num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),2])
            # Get levels
            if vmin >= vmax: print('Wrong order of vmin, vmax in unnormalised posterior plot'); levels = np.linspace(vmax,vmin,num_colors)
            else: levels = np.linspace(vmin,vmax,num_colors)


        # Create figure
        fig = plt.figure(figsize=(10,8))

        if num_plots == 1:
            # Plot countour surface
            im = plt.contourf(self.parameter_mesh[index[0]], self.parameter_mesh[index[1]], Q_hat, levels=levels)

            plt.scatter(self.parameter_mesh[index[0]].flatten()[np.argmax(Q_hat)],self.parameter_mesh[index[1]].flatten()[np.argmax(Q_hat)],label='surface max',marker='x',s=200,color='blue',zorder=10)
            if hasattr(self,'true_parameters'):
                plt.scatter(self.transform_parameters(self.true_parameters,False)[index[0]],self.transform_parameters(self.true_parameters,False)[index[1]],label='Simulation parameter',marker='x',s=100,color='black',zorder=11)
            plt.xlim([np.min(self.parameter_mesh[index[0]]),np.max(self.parameter_mesh[index[0]])])
            plt.ylim([np.min(self.parameter_mesh[index[1]]),np.max(self.parameter_mesh[index[1]])])
            if show_title: plt.title(f'Log unnormalised posterior for {",".join(transformed_parameter_names[i])}')
            plt.xlabel(f'{transformed_parameter_names[i][index[0]]}')
            plt.ylabel(f'{transformed_parameter_names[i][index[1]]}')
            plt.colorbar(im)
            plt.legend(fontsize=10)
        else:

            raise ValueError('generate_log_unnormalised_posteriors_plots with num_plots > 2  does not work.')
            # # multiindex = [[...] for i in range(self.num_learning_parameters+1)]
            # # for i in range(self.num_learning_parameters):
            # #     if i == 0:
            # #         multiindex[i] = list(index)
            # #     elif i not in list(index):
            # #         multiindex[i] = 0
            # # print('multiindex',multiindex)
            #
            # print('index',index)
            # print(np.shape(self.parameter_mesh[list(index)]))
            # print(np.shape(self.parameter_mesh[list(index)][0]))
            # print(np.shape(self.parameter_mesh[list(index)][1]))
            # print(np.shape(np.flip(self.parameter_mesh[list(index)],axis=0)))
            # print(self.parameter_mesh[list(index)])
            # print(np.flip(self.parameter_mesh[list(index)],axis=0))
            #
            #
            # params_mesh = np.meshgrid(*np.flip(self.parameter_mesh[list(index)],axis=0))
            #
            # # Evaluate log unnormalised posterior
            # log_unnormalised_posterior = np.apply_along_axis(self.evaluate_log_posterior, 0, params_mesh[::-1])
            #
            # # Reshape posterior
            # log_unnormalised_posterior = log_unnormalised_posterior.reshape(tuple(parameter_range_lengths))
            #
            #
            # print('np.shape(Q_hat)',np.shape(Q_hat))
            # print('np.shape(self.parameter_mesh)',np.shape(params_mesh))
            # print('np.shape(self.parameter_mesh[index[0]])',np.shape(parameter_mesh[index[0]]))
            # print("np.shape(self.parameter_mesh[index[1]])",np.shape(parameter_mesh[index[1]]))
            #
            # # Plot countour surface
            # im = plt.contourf(parameter_mesh[index[0]], parameter_mesh[index[1]], Q_hat, levels=levels)
            #
            # plt.scatter(self.parameter_mesh[index[0]].flatten()[np.argmax(Q_hat)],self.parameter_mesh[index[1]].flatten()[np.argmax(Q_hat)],label='surface max',marker='x',s=200,color='blue',zorder=10)
            # if hasattr(self,'true_parameters'):
            #     plt.scatter(self.transform_parameters(self.true_parameters,False)[index[0]],self.transform_parameters(self.true_parameters,False)[index[1]],label='Simulation parameter',marker='x',s=100,color='black',zorder=11)
            # plt.xlim([np.min(self.parameter_mesh[index[0]]),np.max(self.parameter_mesh[index[0]])])
            # plt.ylim([np.min(self.parameter_mesh[index[1]]),np.max(self.parameter_mesh[index[1]])])
            # if show_title: plt.title(f'Log unnormalised posterior for {",".join(transformed_parameter_names[i])}')
            # plt.xlabel(f'{transformed_parameter_names[i][index[0]]}')
            # plt.ylabel(f'{transformed_parameter_names[i][index[1]]}')
            # plt.colorbar(im)
            # plt.legend(fontsize=10)
