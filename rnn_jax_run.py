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
from rnn_jax_build import build

#######################################
class rnn_jax_run():
    
    def __init__(self):
        #Task-related paramters
        self.T = 1.0 #Total period
        self.ntime = 100 #number points
        self.dt = self.T/float(self.ntime) #step size
        self.bval = 0.1 #bias
        self.sval = 0.01 #std dev noise
        self.zeros_beginning = 10 #zeros at the start of task before task-related integratio
        self.input_size = 2
        self.output_size = 1
    
        #Training Parameters
        self.state_size = 300 #Network size
        self.g = 1.4 #Chaos
        self.batch_size = 100 
        self.nstep_sizes = 10
        self.num_iters = 100 #Number iterations for each step size
        self.init_step_size = 1e-4 #Initial learning rate
        self.decay_lr = False #Whether to decay learning rate or not
        self.reg_type = 'L2'
        self.reg_size = 2e-6
        
        #Set initial random key
        self.rand_gen_start = 1 #Starting integer for rand generator
        self.run_number = 1 #Current run number
        self.random_key = random.PRNGKey(self.rand_gen_start)
        
        #Set save dir and name
        self.save_dir = "/Users/davidavinci/Documents/MountSinai/Code/PerturbativeRNN/tmp_train"#Directory where to save results
        self.save_name = ''
        self.save_name_task = 'NeuroContextSwitch'
        
        #Instantiate build class
        self.build_instance = build()

    def get_params(self):
        params_dict = dict(bval=self.bval,sval=self.sval,T=self.T,ntime=self.ntime,
                        decay_learning_rate = self.decay_lr,
                        input_size=self.input_size,output_size = self.output_size,
                        state_size = self.state_size,g=self.g, batch_size=self.batch_size,
                        num_iters=self.num_iters, nstep_sizes=self.nstep_sizes,
                        init_step_size=self.init_step_size,run_number = self.run_number,
                        save_dir=self.save_dir,save_name=self.save_name,
                        rand_gen_start = self.rand_gen_start,
                        random_key = self.random_key)
        
        rnn_input_params = self.random_key,self.bval, self.sval, self.T,\
            self.ntime, self.input_size, self.batch_size, self.zeros_beginning, self.save_name_task
            
        rnn_train_params = (self.state_size,self.input_size,self.output_size,\
                            self.g,self.batch_size,self.num_iters,self.nstep_sizes,\
                            self.init_step_size,self.rand_gen_start, self.random_key,\
                            self.decay_lr,self.reg_type,self.reg_size,\
                                self.save_dir, self.save_name)
        return rnn_input_params, rnn_train_params, params_dict
    
    def set_params(self,params_to_tune,run_params):
        input_size,state_size,g,batch_size,num_iters,nstep_sizes,init_step_size,\
            decay_lr = params_to_tune
        
        self.input_size = input_size
        self.state_size = state_size
        self.g = g
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.nstep_sizes = nstep_sizes
        self.init_step_size = init_step_size
        self.decay_lr = decay_lr
        
        run_number,random_key = run_params
        self.run_number = run_number
        self.random_key = random_key
        self.save_name = 'RNN_JAX' + '_' + '_Run%d'%(run_number)#Starts from zero
        
        return
    
    def set_task_params(self,task_params):
        T, ntime, dt, bval, sval, zeros_beginning, input_size, output_size = \
            task_params
        
        self.T = T
        self.ntime = ntime
        self.dt = dt
        self.bval = bval
        self.sval = sval
        self.zeros_beginning = zeros_beginning
        self.input_size = input_size
        self.output_size = output_size
    
        
    def run_training(self,params_to_tune,run_params):

        #Check inputs
        assert len(params_to_tune) == 8, 'Need 8 paramters in params_to_tune'
        assert len(run_params) == 2, 'Need 2 parameters in run_params'
        
        #Set params for current run
        self.set_params(params_to_tune,run_params)
        
        #Get params for training
        rnn_input_params, rnn_train_params, params_dict = self.get_params()
    
        #Check task setup 
        inputs,targets = self.build_instance.build_task(input_params=rnn_input_params)
        assert np.shape(inputs) == (self.ntime,self.batch_size,self.input_size),'Wrong task input shape'
        assert np.shape(targets) == (self.ntime,self.batch_size,self.output_size),'Wrong task output shape'
            
        
        #Save using pickle (faster)
        filehandler = open(self.save_dir+"/RNN_init_params_"+self.save_name+'.pickle',"wb")
        pickle.dump(rnn_train_params,filehandler) 
        pickle.dump(rnn_input_params,filehandler)
        pickle.dump(params_dict,filehandler)
        filehandler.close()
        
        print('Run RNN Training. | Run Nr: %d'%(self.run_number))
        
        #Train RNN
        self.build_instance.train(train_params=rnn_train_params,input_params=rnn_input_params)

    def run_training_loop(self,params_to_tune,total_number_runs):
        
        #Check inputs
        assert len(params_to_tune) == 8, 'Need 8 paramters in params_to_tune'
        assert total_number_runs > 0, 'Total number runs needs to be above 0'
        
        #Generate rand keys. Using different subkeys for different runs
        curr_rand_key = random.PRNGKey(self.rand_gen_start)
        subkeys_runs = random.split(curr_rand_key,total_number_runs)
            
        for run in range(total_number_runs):
            run_params = run,subkeys_runs[run]
            self.run_training(params_to_tune,run_params)

                
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

rnn_run_instance = rnn_jax_run()

rnn_run_instance.run_training_loop(params_to_tune,total_number_runs)   





