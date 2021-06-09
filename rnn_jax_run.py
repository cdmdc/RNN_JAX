#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:46:03 2020

@author: cdmdc

Run RNN
"""
#Import packages
import pickle
import jax.numpy as np
import matplotlib.pyplot as plt

#Import functions
from rnn_jax_build import train
from rnn_jax_task_build import build_task 
from rnn_jax_build import create_rnn_params

def run_training(hyperparams_to_tune,save_name):
    #Params that change
    g,state_size = hyperparams_to_tune
    
    #Input Parameters
    T = 1.0
    ntime = 100
    dt = T/float(ntime)
    bval = 0.1
    sval = 0.001
    input_size = 2
    output_size = 1
    # ntrials = 200
    input_params = (bval, sval, T, ntime)
    
    #Training Parameters
    # state_size = 100
    # g = 0.9
    batch_size = 100
    num_iters = 100
    nstep_sizes = 10
    init_step_size = 0.01
    
    #Check inputs & Targets
    inputs_txexu, targets_txexm = build_task(batch_size, input_params, do_plot=False)
    print("inputs shape: ", inputs_txexu.shape)
    print("targets shape: ", targets_txexm.shape)
    
    #Directory where to save results
    save_dir = "/Users/davidavinci/Documents/MountSinai/Code/ContextSwitchRNN/tmp_train"
    train_params = (state_size,g,batch_size,num_iters,nstep_sizes,
                        init_step_size, save_dir, save_name)

    ############################################################################
    
    #Initialization Parameters
    init_params = create_rnn_params(input_size=input_size, output_size=output_size,
                                    state_size=state_size, g=g)
    
    params_dict = dict(bval=bval,sval=sval,T=T,ntime=ntime,
               input_size=input_size,output_size=output_size,
               state_size = state_size, g=g, batch_size=batch_size,
               num_iters=num_iters, nstep_sizes=nstep_sizes,
               init_step_size=init_step_size, save_dir=save_dir,
               save_name=save_name)
    # #Save using np
    # np.savez(save_dir+"/RNN_init_params_run%d"%(train_count), 
    #         x=init_params,y=train_params)
    
    #Save using pickle (faster)
    filehandler = open(save_dir+'/RNN_init_params_'+save_name,"wb")
    pickle.dump(init_params,filehandler) 
    pickle.dump(train_params,filehandler) 
    pickle.dump(input_params,filehandler)
    pickle.dump(params_dict,filehandler)
    filehandler.close()
    
    # print('Run Training. | g param %.2f | state size param %.2f |'%(g,state_size))
    
    #Train RNN
    train(train_params=train_params,input_params=input_params,init_params=init_params)

############## RUN TRAINING #####################################    


g = 1.0; state_size=20; train_counter = 1
print('Training | g=%.2f | state_size=%.2f | Run # %d',g,state_size,train_counter)
hyperparams_to_tune = (g,state_size)
save_name = 'g=%d_statesize=%d_LRhigh_RunNr%d'%(g,state_size,train_counter)
run_training(hyperparams_to_tune=hyperparams_to_tune,save_name=save_name)
    