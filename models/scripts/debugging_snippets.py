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
