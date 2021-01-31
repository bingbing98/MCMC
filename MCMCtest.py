import numpy as np
import math
import random
import matplotlib.pyplot as plt
'''
true model:
    x^2 = y
data:
    y = 1
true result:
    x = 1 and x = -1

use MCMC to sample the result
'''
# set parameters
model_space = [-6,6]
sigma       = 0.4
step_size   = 0.1
thre_num    = 1000
accept_num  = 5000
fit_data    = 1
#===============================================#
def forward(model):
    syndata = model**2
    return syndata

def cal_err(syndata,true_data):
    err = (syndata-true_data)**2
    return err

def cal_prob(err1,err2):
    lm1 = math.exp(-err1/(2*sigma**2))
    lm2 = math.exp(-err2/(2*sigma**2))
    p_accept = min(lm2/lm1,1)
    return p_accept

def resample(old_model):
    new_model = old_model+random.gauss(0,step_size)
    new_model = min(max(model_space),new_model)
    new_model = max(min(model_space),new_model)
    return new_model


init_model  = random.uniform(min(model_space),max(model_space))

old_model   = init_model
syn_data    = forward(old_model)
err1        = cal_err(syn_data,fit_data)
step        = 0
err         = np.zeros(thre_num+accept_num)
sam_result  = np.zeros(accept_num)

while step < thre_num+accept_num:
    next_move   = resample(old_model) 
    syn_data    = forward(next_move)
    err2        = cal_err(syn_data,fit_data)
    
    u = random.uniform(0,1)
    p = cal_prob(err1,err2)
    if u <= p:
        err1 = err2
        old_model = next_move

        if step > thre_num:
            sam_result[step-thre_num]=old_model

        err[step]=err1
        step += 1

plt.figure()
plt.subplot(2,1,1)
plt.hist(sam_result,label='result distribution')
plt.legend()
plt.subplot(2,1,2)
plt.plot(err,label='Misfit')
plt.legend()
plt.show()
