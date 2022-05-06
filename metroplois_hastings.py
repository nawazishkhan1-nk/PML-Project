import scipy
from scipy.stats import norm
import numpy as np
import shapeworks as sw
import shapeworks_py
import sampling_methods
import math

#  Metropolis algorithm (MCMC), provides samples from the evaluator distribution by drawing from generator and stochastic accept/reject decisions
#  generator needs to be symmetric

class Metropolis:
    def __init__(self, generator, evaluator, logger) -> None:
        self.generator = generator
        self.logger = logger
        self.evaluator = evaluator
    
    def get_next_sample(self, current):
        currentP = self.evaluator.logValue(current)
        proposal = self.generator.propose(current)
        proposalP = self.evaluator.logValue(proposal)
        # acceptance probability
        a = proposalP - currentP
        # accept or reject
        check_val = norm(loc = 0, scale = 1).rvs(1)
        if (a > 0.0 or check_val< np.exp(a)):
            self.logger.accept(current, proposal, self.generator, self.evaluator)
        else:
            self.logger.reject(current, proposal, self.generator, self.evaluator)

transition_model = lambda x: np.random.normal(x,[0.05,5],(2,))
def prior(w):
    if(w[0]<=0 or w[1] <=0):
        return 0
    else:
        return 1
    
def manual_log_lik_gamma(x,data):
    return np.sum((x[0]-1)*np.log(data) - (1/x[1])*data - x[0]*np.log(x[1]) - np.log(math.gamma(x[0])))
def log_lik_gamma(x,data):
    return np.sum(np.log(scipy.stats.gamma(a=x[0],scale=x[1],loc=0).pdf(data)))
def metropolis_hastings(likelihood_computer,prior, transition_model, param_init,iterations,data,acceptance_rule):
    x = param_init
    accepted = []
    rejected = []   
    for i in range(iterations):
        x_new =  transition_model(x)    
        x_lik = likelihood_computer(x,data)
        x_new_lik = likelihood_computer(x_new,data) 
        if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)                   
    return np.array(accepted), np.array(rejected)
def acceptance(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        return (accept < (np.exp(x_new-x)))

accepted, rejected = metropolis_hastings(manual_log_lik_gamma,prior,transition_model,[4, 10], 50000,transition_model,acceptance)