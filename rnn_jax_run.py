#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:46:03 2020

@author: davidavinci

Class to run RNN built in JAX

Dependencies: jax.numpy, jax.random, rnn_jax_build, pickle

Class to run RNN training on Context Switch task. However, by overriding
build_task in rnn_jax_build one can plug in custom task building function.

Class is initialized with parameters that work well for neural tasks such as 
the context switch task. However, these parameters can be overriden/tuned 
for individual runs using set_params

Task related variables can be set using set_task_params after initializing class

"""

import pickle
import jax.numpy as np
from jax import random #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers


#Import train class rnn_jax_build
from rnn_jax_build import rnn_run

#######################################

                
########################## RUN TRAINING #################################    

total_number_runs = 1

#Param settings 
input_size = 2
state_size = 100
g = 1.5
batch_size = 100
num_iters = 100
nstep_sizes = 10
init_step_size = 1e-3
decay_lr = False

params_to_tune = input_size,state_size,g,batch_size,\
    num_iters,nstep_sizes,init_step_size,decay_lr

rnn_run_instance = rnn_run()

rnn_run_instance.run_training_loop(params_to_tune,total_number_runs)   





