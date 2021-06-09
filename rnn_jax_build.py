#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:25:59 2020

@author: cdmdc
"""
#Import packages
from __future__ import absolute_import
from __future__ import print_function
import jax.numpy as np
from jax import random #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers
from jax import jit,grad
import numpy.random as npr
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental.optimizers import adam
import pickle

#Import task builder function (switchable)
from rnn_task_context_switch_build import build_task

########################################################################
def concat_multiply(weights, *args):
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    return np.dot(cat_state, weights)

def create_rnn_params(input_size, state_size, output_size,
                      g=1.0, rs=npr.RandomState(0)):

    input_factor = 1.0 / np.sqrt(input_size)
    hidden_scale = 1.0
    hidden_factor = g / np.sqrt(state_size + 1)
    predict_factor = 1.0 / np.sqrt(state_size + 1)
    return {'hidden unit': rs.randn(1, state_size) * hidden_scale,
            'change':       np.concatenate((rs.randn(input_size, state_size) * input_factor,
                                            rs.randn(state_size + 1, state_size) * hidden_factor),
                                           axis=0),
            'predict':      rs.randn(state_size + 1, output_size) * predict_factor}

def rnn_predict(params, inputs, return_hiddens=False):
    def update_rnn(input_item, hidden_units):
        return np.tanh(concat_multiply(params['change'], input_item, hidden_units))

    def hidden_to_output_probs(hidden_units):
        return concat_multiply(params['predict'], hidden_units)

    batch_size = inputs.shape[1]
    hidden_units = np.repeat(params['hidden unit'], batch_size, axis=0)
    outputs_time = []
    hiddens_time = []
    for input_item in inputs:  # Iterate over time steps.
        hidden_units = update_rnn(input_item, hidden_units)
        outputs_time.append(hidden_to_output_probs(hidden_units))
        if return_hiddens: hiddens_time.append(hidden_units)
    return np.array(outputs_time), np.array(hiddens_time)

def rnn_mse(params, inputs, targets):
    outputs_time, _ = rnn_predict(params, inputs)
    return np.mean((outputs_time - targets)**2)


def train(train_params,input_params, init_params=None):
    
    state_size,g,batch_size,num_iters,nstep_sizes,\
        init_step_size, save_dir, save_name = train_params
        
    # Allow continued training.
    if init_params is None:
        init_params = create_rnn_params(input_size=2, output_size=1,
                                        state_size=state_size, 
                                        g=g)
        
    def training_loss(params, iter):
        inputs,targets = build_task(batch_size,input_params)
        mse = rnn_mse(params, inputs, targets)
        l2_reg = 2e-6
        reg_loss = l2_reg * np.sum(params['change']**2)
        return mse + reg_loss        
              
    #Get optimizer & define update step
    step_sizes = iter([init_step_size*(0.333333**n) for n in range(nstep_sizes)])
    opt_init, opt_update, get_params = adam(step_size=next(step_sizes),
                                            b1=0.9, b2=0.999, eps=1e-8)
    trained_params = init_params
    opt_state = opt_init(init_params)
    
    @jit
    def step(j,opt_state,batch):
        params = get_params(opt_state)
        g = grad(training_loss)(params,j)#build gradient of loss function using jax grad
        return opt_update(j, g, opt_state)
    
    loss_log = []
    for lr_step,step_size in enumerate(step_sizes):
        print("(%d/%d) Training RNN for %d steps at step size %f" % (lr_step+1, nstep_sizes, num_iters, step_size))
        for iter_step in range(num_iters):
            opt_state = step(lr_step,opt_state,trained_params)#Run optimization step
            if iter_step % 10 == 0:#Print out current loss
                curr_loss = training_loss(get_params(opt_state), 0)
                loss_log.append(curr_loss)
                print("Iteration: ", iter_step, " Train loss:", curr_loss)
         
        #Get temp results at step_size change & save    
        trained_params = get_params(opt_state)
        loss_log_array = np.array(loss_log)
        # #Save using np
        # np.savez(save_dir+"/RNN_trained_params_run%d"%(train_count), x=trained_params)
        
        #Save using pickle (faster)
        filehandler = open(save_dir+'/RNN_trained_params_'+save_name,"wb")
        pickle.dump(trained_params,filehandler)
        pickle.dump(loss_log_array, filehandler)
        filehandler.close()
  
    print("Le Fin.")
    return trained_params