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

Original run_MCMC_Gaussian:
Pure MH samples = 9999 Pure MH Acceptance efficiency =  0.2007
Diffusion MCMC Chain Started
Number of retrains = 48/100
Diffusion acceptance efficiency = 0.5263157894736842
Metropolis acceptance efficiency = 0.20987654320987653
Previous Retrain Time = 8.0 seconds 
Total Retrain Time = 195.0 seconds 

Modified:
