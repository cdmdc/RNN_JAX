#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 08:42:02 2021

@author: root
"""


import jax.numpy as np
from jax import jit,grad,ops
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental.optimizers import adam,sm3,adagrad,make_schedule,optimizer
import pickle
# from scipy import signal


# Generate random key
from jax import random #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers

#Packages relevant for build_task
import numpy
from scipy import signal


################## HELPER TRAIN FUNS ################################
# Training loss is the negative log-likelihood of the training examples.
    
class rnn_run():
    
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

class build():
    
    def __init__(self):
        return
    
    def concat_multiply(self,weights, *args):
        cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
        return np.dot(cat_state, weights)
    
    def concat_args(self,*args):
        return np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    
    def rnn_mse(self,params, inputs, targets):
        outputs_time, _ = self.rnn_predict(params, inputs)
        return np.mean((outputs_time - targets)**2)
    
    def rnn_predict(self,params, inputs, return_hiddens=False):
        def update_rnn(input_item, hidden_units):
            return np.tanh(self.concat_multiply(params['change'], input_item, hidden_units))
    
        def hidden_to_output_probs(hidden_units):
            return self.concat_multiply(params['predict'], hidden_units)
    
        batch_size = inputs.shape[1]
        hidden_units = np.repeat(params['hidden unit'], batch_size, axis=0)
        outputs_time = []#outputs
        hiddens_time = []#hidden activations over time
        for input_item in inputs:  # Iterate over time steps.
            hidden_units = update_rnn(input_item, hidden_units)
            outputs_time.append(hidden_to_output_probs(hidden_units))
            if return_hiddens: hiddens_time.append(hidden_units)
        return np.array(outputs_time), np.array(hiddens_time)
    
    
    def create_rnn_params(self,input_size, state_size, output_size,
                          g, random_key):
    
        input_factor = 1.0 / np.sqrt(input_size)
        hidden_scale = 1.0
        hidden_factor = g / np.sqrt(state_size+1)
        predict_factor = 1.0 / np.sqrt(state_size+1)
        return {'hidden unit': np.array(random.normal(random_key,(1, state_size)) * hidden_scale),
                'change':       np.concatenate((random.normal(random_key,(input_size, state_size)) * input_factor,
                                                random.normal(random_key,(state_size+1, state_size)) * hidden_factor),
                                               axis=0),#hidden weights
                'predict':      random.normal(random_key,(state_size+1, output_size)) * predict_factor}#readout weights
    
    def rnn_neglikelihood(self,params, inputs, targets):
        preds, _ = self.rnn_predict(params, inputs)
        label_probs = preds * targets + (1 - preds) * (1 - targets)
        return -np.sum(np.log(label_probs))
    
    #Build Context-Switch Task
    def build_task(self,input_params):
        random_key, bval, sval, T, ntime, num_inputs, batch_size,\
            zeros_beginning, save_name_tasks = input_params
            
        ntrials = np.asarray(batch_size/2,dtype=int)
        dt = T/float(ntime)
        
        # biases_1xexw = np.expand_dims(bias_val * 2.0 * (numpy.random.rand(ntrials,nwninputs) -0.5), axis=0)
        stddev = sval / np.sqrt(dt)
        noise_input = stddev * numpy.random.randn(ntime-zeros_beginning, ntrials, num_inputs)
        noise_input_pluszero = np.concatenate((np.zeros((zeros_beginning,ntrials,num_inputs)),
                                      noise_input))#Allow context to establish
        
        square_wave_input = np.concatenate((np.zeros((numpy.int(ntime*(2/5)),ntrials,1)),
                                            np.ones((numpy.int(ntime*(1/5)),ntrials,1)),
                                            np.zeros((numpy.int(ntime*(2/5)),ntrials,1))),
                                           axis=0)
        
        sqwave_plus_noise = np.concatenate((square_wave_input,
                                            -1*square_wave_input),axis=2)+ noise_input_pluszero
        
        context_1 = np.expand_dims(numpy.random.randint(0,2, ntrials), axis=1)
        context_2 = np.concatenate((context_1.astype(np.float64), 
                                      np.logical_not(context_1).astype(np.float64)), 
                                     axis=1)
        context_12 = np.repeat(np.expand_dims(context_2, axis=0), ntime, axis=0)
        
        
        gaussian_window = signal.gaussian(numpy.int(ntime*(1/10)), std=7)
        target_pulse = np.ones((numpy.int(ntime*(1/10)),))*gaussian_window
        target_pulse_m = np.repeat(np.expand_dims(target_pulse,axis=1),ntrials,axis=1)
        target_pulse_m2 = np.expand_dims(target_pulse_m,axis=2)
        targets = np.concatenate((np.zeros((numpy.int(ntime*(6/10)),ntrials,1)),
                                            target_pulse_m2,
                                            np.zeros((numpy.int(ntime*(3/10)),ntrials,1))),
                                           axis=0)
        
        noise_output = stddev * numpy.random.randn(ntime-zeros_beginning, ntrials, num_inputs)
        noise_output_pluszero = np.concatenate((np.zeros((zeros_beginning,ntrials,num_inputs)),
                                      noise_output))#Allow context to establish
        targets_1 = np.where(context_12[:,:,0],targets[:,:,0],np.zeros((ntime,ntrials))+noise_output_pluszero[:,:,0])
        targets_2 = np.where(context_12[:,:,1],np.zeros((ntime,ntrials))+noise_output_pluszero[:,:,0],targets[:,:,0])
        
        context_total = np.expand_dims(np.concatenate((context_12[:,:,0],
                                                       context_12[:,:,1]),axis=1),axis=2)
        inputs_total = np.expand_dims(np.concatenate((sqwave_plus_noise[:,:,0], 
                                                      sqwave_plus_noise[:,:,1]),axis=1),axis=2)
        inputs_total_plus_context = np.concatenate((inputs_total,context_total),axis=2)
        targets_total = np.expand_dims(np.concatenate((targets_1,targets_2),axis=1),axis=2)
        
            
        return inputs_total_plus_context, targets_total
    
    ############# TRAIN FUNS #################################
    
    def train(self,train_params,input_params, init_params=None):
    
        state_size,input_size,output_size,g,batch_size,num_iters,nstep_sizes,\
            init_step_size, rand_gen_start, random_key, decay_lr,\
            reg_type, reg_size, save_dir, save_name = train_params
        
        # Allow continued training.
        if init_params is None:
            #Use first rand key to draw initial parameters
            init_params = self.create_rnn_params(input_size=input_size, 
                                            output_size=output_size,
                                            state_size=state_size,
                                            g=g,random_key=random_key)
            
    
        def training_loss(params, iter):
            inputs,targets = self.build_task(input_params)
            mse = self.rnn_mse(params, inputs, targets)
            
            if reg_type == 'L1':
                reg_loss = reg_size * np.abs(np.sum(params['change']))
            elif reg_type == 'L2':
                reg_loss = reg_size * np.sum(params['change']**2)
            return mse + reg_loss
                  
        #Get optimizer & define update step
        if decay_lr:
            step_sizes = iter([init_step_size*(0.333333**n) for n in range(nstep_sizes)])
        else:
            step_sizes = iter([init_step_size*(1**n) for n in range(nstep_sizes)])
        opt_init, opt_update, get_params = adam(step_size=next(step_sizes),
                                                b1=0.9, b2=0.999, eps=1e-8)
        trained_params = init_params
        opt_state = opt_init(init_params) 
        
        
        @jit
        def step(i,opt_state,batch):
            params = get_params(opt_state)
            g = grad(training_loss)(params,i)#build gradient of loss function 
            return opt_update(i, g, opt_state)
        
        loss_log = []
        rand_gen_counter = 0
        for i,step_size in enumerate(step_sizes):
            print("(%d/%d) Training RNN for %d steps at step size %f" % (i+1, nstep_sizes, num_iters, step_size))
            for j in range(num_iters):
                opt_state = step(i,opt_state,trained_params)#Run optimization step
                if j % 10 == 0:#Print out current loss
                    curr_loss = training_loss(get_params(opt_state), 0)
                    loss_log.append(curr_loss)
                    print("Iteration: ", j, " Train loss:", curr_loss)
                rand_gen_counter = rand_gen_counter + 1 #Get new random seed for each batch
             
            #Get temp results at step_size change & save    
            trained_params = get_params(opt_state)
            loss_log_array = np.array(loss_log)
            
            #Save trained params at each step size change
            filehandler = open(save_dir+"/RNN_trained_params_"+save_name+'.pickle',"wb")
            pickle.dump(trained_params,filehandler)
            pickle.dump(loss_log_array, filehandler)
            filehandler.close()        
            # np.save(save_dir+"/RNN_trained_params_run%d"%(train_count), \
            #         trained_params,loss_log_array)
        print("Done.")
        return trained_params
