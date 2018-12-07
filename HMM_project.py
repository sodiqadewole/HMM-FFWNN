# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:21:26 2018

@author: SODIQ-PC
"""

from hmmlearn import hmm
import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score


# generate data from a binary distribution
num_states = 2
window = 2
samples = 1000000
    


# get sample slice of the generated data
x = gen_data(states, 5, samples)
y_pred = hmm_simulate(num_states, window, x)

y_true = x[2:,]
   
# compare prediction with the ground truth samples
# calculate the accuracy of the hmm
hmm_accuracy = accuracy_score(y_true, y_pred)
    

def gen_data(num_states, k, n):
    ''' generate sample 2D arry data as sequence of observations'''
    generate_ = np.random.multinomial(num_states, [1/k]*k, n)
    
    data = [[[i] for i in row] for row in generate_]
    data = np.array(np.concatenate(data))   
    
    return data


def hmm_simulate(num_states, window, x):
    '''fit model to data samples generated'''
    
    # Concatenate the array into a 1D array
    # reshape the 1D array into a feature column
    x = x.reshape(-1,1)
    
    # instantiate an HMM with discrete multinomial emissions
    model = hmm.MultinomialHMM(n_components=num_states)

    # fit the model to estimate the model parameters
    model.fit(x)
    
    #Find the most likely state sequence corresponding to observations
    states_seq = model.predict(x) 
    
    transmat = model.transmat_
    
    # get the cumulative cdf of the transition probability matrix across col
    trans_cdf = np.cumsum(model.transmat_, axis=1)
    
    y_pred = np.array([])
    
    # Initialize a random state generator
    random_state = check_random_state(model.random_state)
    
    for i in range(x.shape[0]-2):
        
        # the next state is the state with the maximum likelihood
        next_state = (trans_cdf[states_seq[i+window]] > random_state.rand()).argmax()
    
        # generate next observation from the next state
        next_obs = model._generate_sample_from_state(next_state, random_state)
        
        y_pred = np.append(y_pred, next_obs)
    
    return y_pred

    
    