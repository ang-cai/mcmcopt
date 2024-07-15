# MCMC-diffusion (Optimized)
A Metropolis-Hastings MCMC sampler accelerated via diffusion models. Please cite https://arxiv.org/abs/2309.01454 if you use this code in your research.

Optimization ideas from Ewan Cameron at Telethon Kids Insitute.

Replace line 158 with:
 
*a copy-paste of current lines 103-131*
Then
QMH = scipy.stats.multivariate_normal(theta_prime-theta,0,self.self.sigma)
Q_ratio = (self.diffusion_prob*Q+(1-self.diffusion_prob)*QMH)/(self.diffusion_prob*Q_prime+(1-self.diffusion_prob)*QMH)
 
a = min(1, self.log_likelihood(theta_prime,self.dim)self.log_likelihood(theta,self.dim)*Q_ratio)
 
Remove lines 116-117
 
Replace line 135 with:
 
QMH = scipy.stats.multivariate_normal(theta_prime-theta,0,self.self.sigma)
Q_ratio = (self.diffusion_prob*Q+(1-self.diffusion_prob)*QMH)/(self.diffusion_prob*Q_prime+(1-self.diffusion_prob)*QMH)
