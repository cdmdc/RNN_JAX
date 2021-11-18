#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 08:42:02 2021

@author: root

Inherits from build_class in rnn_jax_build, but overrides build function
with a task builder function that allows for creating nine different
neuroscience tasks (as in Duncker et al. NeurIPS 2020).

Also adds chain_tasks which allows chaining tasks together and training 
them sequentially.

"""


import jax.numpy as np
from jax import jit,grad,ops
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental.optimizers import adam
import pickle
# from scipy import signal


# Generate random key
from jax import random #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers

from rnn_jax_build import build, rnn_run


################## HELPER TRAIN FUNS ################################
# Training loss is the negative log-likelihood of the training examples.

class build_chaintasks(build):
    
    def __init__(self):
        super().__init__(self)
        return
    
    def build_task(self,input_params,task_select):
        """
        Use dt=1 and ntime=100
        """
        random_key, bval, sval, T, ntime, num_inputs, batch_size,\
        zeros_beginning, save_name_tasks = input_params
                
        #Generate a particular new random key for a batch to get fresh batch
        batch_key = random_key
        num_batch_subkeys = 15
        subkeys = random.split(batch_key,num_batch_subkeys)
        
        dt = T/float(ntime)
        stddev = sval / np.sqrt(dt)
        ntrials = batch_size
        
        # _,subkey = random.split(key)#draw new key each time you draw new noise
        def create_angle_vecs(angles,rkey):
            angle_rand_idx = random.randint(rkey,(ntrials,),0,np.size(angles))
            cos_angles = np.array([np.cos(np.radians(angles[i])) for i in angle_rand_idx])
            sin_angles = np.array([np.sin(np.radians(angles[i])) for i in angle_rand_idx])
            cos_angles = np.expand_dims(cos_angles,axis=0)
            cos_angles = np.expand_dims(cos_angles,axis=2)
            sin_angles = np.expand_dims(sin_angles,axis=0)
            sin_angles = np.expand_dims(sin_angles,axis=2)
            return cos_angles,sin_angles
    
        angles = np.arange(0,360,10)
        cos_angles,sin_angles = create_angle_vecs(angles,subkeys[0])    
        
        #Opposite angles. Use same random key to create matching angles
        angles_opposite = angles + 180
        cos_angles_opposite,sin_angles_opposite = create_angle_vecs(angles_opposite,subkeys[0]) 
        
        def draw_noise_input(ntime,zeros_beginning,ntrials,num_inputs,stddev,rkey):
            noise_input = stddev * random.normal(rkey,(ntime, ntrials, num_inputs))
            noise_input_pluszero = np.concatenate((np.zeros((zeros_beginning,ntrials,num_inputs)),
                                      noise_input))#Allow context to establish 
            return noise_input_pluszero
        
        ##### FIXATION INPUT
        input_f = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                  np.ones((np.int_(ntime*3/4),ntrials,1)),
                                  np.zeros((np.int_(ntime*(1/4)),ntrials,1))),axis=0)
        
        inputs_f_plusnoise = input_f + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev, subkeys[1])
    
        ##### RULE INPUT
        input_r_zero = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                  np.zeros((ntime,ntrials,1))),axis=0)
        input_r_zero_plusnoise = input_r_zero + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev,subkeys[1])
        input_r_one = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                  np.ones((ntime,ntrials,1))),axis=0)
        input_r_one_plusnoise = input_r_one + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev,subkeys[1]) 
        inputs_r = (input_r_zero_plusnoise,input_r_one_plusnoise)
        
        if task_select == 'delay_pro' or task_select == 'delay_anti' or task_select == 'mem_pro' or task_select == 'mem_anti':
            
        
            ##### STIMULUS INPUT
            input_s_cos = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                      np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                      np.multiply(np.repeat(cos_angles,np.int_(ntime*3/4),axis=0),
                                                  np.ones((np.int_(ntime*3/4),ntrials,1)))),axis=0)
            input_s_cos_plusnoise = input_s_cos + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev,subkeys[2])#
            
            input_s_sin = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                      np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                      np.multiply(np.repeat(sin_angles,np.int_(ntime*3/4),axis=0),
                                                  np.ones((np.int_(ntime*3/4),ntrials,1)))),axis=0)  
            input_s_sin_plusnoise = input_s_sin + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev,subkeys[3])#3    
            
            inputs_s = (input_s_cos_plusnoise,input_s_sin_plusnoise)
            
            
            ##### MEMORY INPUT
            # duration_val_stim = np.int_(20)#np.int_(ntime*random.uniform(subkeys[1],(1,),minval=1/8,maxval=4/8)[0])
            # duration_val_delay = np.int_(ntime - duration_val_stim - 3*np.int_(ntime*1/4))
            duration_val_stim = np.int_(ntime*2/4)
            duration_val_delay = np.int_(ntime*0)
            
            input_s_cos_mem = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                      np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                      np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),#np.int_(ntime*2/4)
                                                  np.ones((duration_val_stim,ntrials,1))),
                                      np.zeros((duration_val_delay,ntrials,1)),
                                      np.zeros((np.int_(ntime*1/4),ntrials,1))),axis=0)
            input_s_cos_mem_plusnoise = input_s_cos_mem + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev,subkeys[4])#2
            
            input_s_sin_mem = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                      np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                      np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                                  np.ones((duration_val_stim,ntrials,1))),
                                      np.zeros((duration_val_delay,ntrials,1)),
                                      np.zeros((np.int_(ntime*1/4),ntrials,1))),axis=0)  
            input_s_sin_mem_plusnoise = input_s_sin_mem + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev,subkeys[5])  
        
            inputs_s_mem = (input_s_cos_mem_plusnoise,input_s_sin_mem_plusnoise)
        
            ###### GENERAL TASK TARGETS
            target_cos = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                         np.zeros((np.int_(3/4*ntime),ntrials,1)),
                                         np.multiply(np.repeat(cos_angles,np.int_(1/4*ntime),axis=0),
                                                     np.ones((np.int_(1/4*ntime),ntrials,1)))),axis=0)
            target_sin = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                         np.zeros((np.int_(3/4*ntime),ntrials,1)),
                                         np.multiply(np.repeat(sin_angles,np.int_(1/4*ntime),axis=0),
                                                     np.ones((np.int_(1/4*ntime),ntrials,1)))),axis=0)    
            targets_s = (target_cos,target_sin)
               
            
            # targets_ones_mem = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
            #                              np.zeros((np.int_(3/4*ntime),ntrials,1)),
            #                              np.ones((np.int_(1/4*ntime),ntrials,1))),axis=0)     
            
            #Opposite direction
            target_cos_opposite = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                         np.zeros((np.int_(3/4*ntime),ntrials,1)),
                                         np.multiply(np.repeat(cos_angles_opposite,np.int_(1/4*ntime),axis=0),
                                                     np.ones((np.int_(1/4*ntime),ntrials,1)))),axis=0)
            target_sin_opposite = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                         np.zeros((np.int_(3/4*ntime),ntrials,1)),
                                         np.multiply(np.repeat(sin_angles_opposite,np.int_(1/4*ntime),axis=0),
                                                     np.ones((np.int_(1/4*ntime),ntrials,1)))),axis=0)    
            targets_opposite = (target_cos_opposite,target_sin_opposite)                
        
    
        elif task_select == 'mem_dm1' or task_select == 'mem_dm2' or task_select == 'context_mem_dm1' or task_select == 'context_mem_dm2' or task_select == 'multi_mem':
            #Shows second stim after delay, output stronger stim
            # angles2 = np.arange(0,180,10)
            cos_angles2,sin_angles2 = create_angle_vecs(angles,subkeys[10])  
            target_vals_cos = np.where(cos_angles2>=cos_angles,cos_angles2,cos_angles)
            target_vals_sin = np.where(sin_angles2>=sin_angles,sin_angles2,sin_angles)
            
            #Draw a particular delay duration for current batch
            duration_val_stim = np.int_(10)#
            duration_val_interstim = 2*duration_val_stim
            duration_val_delay = ntime - 2*duration_val_stim - duration_val_interstim - 2*np.int_(ntime*1/4)
    
            
             ##### MEMORY INPUT
            input_s_cos_mem_short = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                  np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                  np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),
                                              np.ones((duration_val_stim,ntrials,1))),
                                  np.zeros((duration_val_interstim,ntrials,1)),
                                  np.multiply(np.repeat(cos_angles2,duration_val_stim,axis=0),
                                  np.ones((duration_val_stim,ntrials,1))),
                                  np.zeros((duration_val_delay,ntrials,1)),
                                  np.zeros((np.int_(ntime*1/4),ntrials,1))),axis=0)
            input_s_cos_mem_short_plusnoise = input_s_cos_mem_short + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev,subkeys[8])#8
            
            input_s_sin_mem_short = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                  np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                  np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                              np.ones((duration_val_stim,ntrials,1))),
                                  np.zeros((duration_val_interstim,ntrials,1)),
                                  np.multiply(np.repeat(sin_angles2,duration_val_stim,axis=0),
                                  np.ones((duration_val_stim,ntrials,1))),
                                  np.zeros((duration_val_delay,ntrials,1)),
                                  np.zeros((np.int_(ntime*1/4),ntrials,1))),axis=0)
            input_s_sin_mem_short_plusnoise = input_s_sin_mem_short + draw_noise_input(ntime, zeros_beginning, ntrials, 1, stddev,subkeys[9])#9
            
            inputs_s_mem_dm = (input_s_cos_mem_short_plusnoise,input_s_sin_mem_short_plusnoise)
                        
              
            ###### GENERAL TASK TARGETS
            target_zeros = ntime - 2*np.int_(ntime*1/4)
            target_s_cos_mem_long = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                                    np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                                    np.zeros((target_zeros,ntrials,1)),
                                                    np.multiply(np.repeat(target_vals_cos,np.int_(ntime*1/4),axis=0),
                                                                np.ones((np.int_(ntime*1/4),ntrials,1)))),axis=0)
            target_s_sin_mem_long = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                                    np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                                    np.zeros((target_zeros,ntrials,1)),
                                                    np.multiply(np.repeat(target_vals_sin,np.int_(ntime*1/4),axis=0),
                                                                np.ones((np.int_(ntime*1/4),ntrials,1)))),axis=0)
            
            target_vals_multi_mem = np.where(cos_angles2+sin_angles2>=cos_angles+sin_angles,
                                             cos_angles2+sin_angles2,cos_angles+sin_angles)
            targets_s_multi_mem = np.concatenate((np.zeros((zeros_beginning,ntrials,1)),
                                                    np.zeros((np.int_(ntime*1/4),ntrials,1)),
                                                    np.zeros((target_zeros,ntrials,1)),
                                                    np.multiply(np.repeat(target_vals_multi_mem,np.int_(ntime*1/4),axis=0),
                                                                np.ones((np.int_(ntime*1/4),ntrials,1)))),axis=0) 
            targets_s_mem_dm = (target_s_cos_mem_long,target_s_sin_mem_long,targets_s_multi_mem)
    
    
        ###### FULL TASKS
            
        if task_select == 'delay_pro':
            inputs = np.concatenate((inputs_f_plusnoise,
                                        inputs_s[0],
                                        inputs_s[1],#),axis=2)
                                        inputs_r[1],
                                        inputs_r[0],
                                        inputs_r[0],
                                        inputs_r[0]),axis=2)
            targets = np.concatenate((input_f,
                                        targets_s[0],
                                        targets_s[1]),axis=2)
        elif task_select == 'delay_anti':
            inputs = np.concatenate((inputs_f_plusnoise,
                                        inputs_s[0],
                                        inputs_s[1],#),axis=2)
                                        inputs_r[0],
                                        inputs_r[1],
                                        inputs_r[0],
                                        inputs_r[0]),axis=2)
            targets = np.concatenate((input_f,
                                        targets_opposite[0],
                                        targets_opposite[1]),axis=2)  
        elif task_select == 'mem_pro':
            inputs = np.concatenate((inputs_f_plusnoise,
                                      inputs_s_mem[0],
                                      inputs_s_mem[1],#),axis=2)
                                        inputs_r[0],
                                        inputs_r[0],
                                        inputs_r[1],
                                        inputs_r[0]),axis=2)
            targets = np.concatenate((input_f,
                                      targets_s[0],
                                      targets_s[1]),axis=2)
        elif task_select == 'mem_anti':
            inputs = np.concatenate((inputs_f_plusnoise,
                                        inputs_s_mem[0],
                                        inputs_s_mem[1],#),axis=2)
                                        inputs_r[0],
                                        inputs_r[0],
                                        inputs_r[0],
                                        inputs_r[1]),axis=2)
            targets = np.concatenate((input_f,
                                        targets_opposite[0],
                                        targets_opposite[1]),axis=2)  
        elif task_select == 'mem_dm1':
            inputs = np.concatenate((inputs_f_plusnoise,
                                        inputs_s_mem_dm[0],#),axis=2)
                                        inputs_r[0],
                                        inputs_r[1],
                                        inputs_r[1],
                                        inputs_r[1]),axis=2)
            targets = np.concatenate((input_f,
                                        targets_s_mem_dm[0]),axis=2) 
        elif task_select == 'mem_dm2':
            inputs = np.concatenate((inputs_f_plusnoise,
                                        inputs_s_mem_dm[1],#),axis=2)
                                        inputs_r[1],
                                        inputs_r[0],
                                        inputs_r[1],
                                        inputs_r[1]),axis=2)
            targets = np.concatenate((input_f,
                                        targets_s_mem_dm[1]),axis=2)
        elif task_select == 'context_mem_dm1':
            inputs = np.concatenate((inputs_f_plusnoise,
                                       inputs_s_mem_dm[0],
                                        inputs_s_mem_dm[1],#),axis=2)
                                        inputs_r[1],
                                        inputs_r[1],
                                        inputs_r[0],
                                        inputs_r[1]),axis=2)
            targets= np.concatenate((input_f,targets_s_mem_dm[0]),axis=2)
        elif task_select == 'context_mem_dm2':
            inputs = np.concatenate((inputs_f_plusnoise,
                                       inputs_s_mem_dm[0],
                                        inputs_s_mem_dm[1],#),axis=2)
                                        inputs_r[1],
                                        inputs_r[1],
                                        inputs_r[1],
                                        inputs_r[0]),axis=2)
            targets = np.concatenate((input_f,targets_s_mem_dm[1]),axis=2)
        elif task_select == 'multi_mem':
            inputs = np.concatenate((inputs_f_plusnoise,
                                       inputs_s_mem_dm[0],
                                        inputs_s_mem_dm[1],#),axis=2)
                                        inputs_r[1],
                                        inputs_r[1],
                                        inputs_r[1],
                                        inputs_r[1]),axis=2)
            targets = np.concatenate((input_f,targets_s_mem_dm[2]),axis=2)

                
        inputs = np.array(inputs)
        targets = np.array(targets)
    
        #Shuffle elements
        shuffle_idx = random.randint(subkeys[10],(ntrials,),0,ntrials)
        inputs = inputs[:,shuffle_idx,:]
        targets = targets[:,shuffle_idx,:]
            
        return inputs, targets
    
    def chain_tasks(self,input_params,tasks_to_chain):
        '''
        For now chains Mem_Pro and Mem_Anti.
        Need to check that input/output size of tasks are the same
        '''
        random_key, bval, sval, T, ntime, num_inputs, batch_size,\
        zeros_beginning, save_name_tasks = input_params
        
        inputs_chained = []
        targets_chained = []
        
        for task in tasks_to_chain:
            inputs_task,targets_task = self.build_task(input_params,task)
            inputs_chained.append(inputs_task)
            targets_chained.append(targets_task)
        
        inputs_chained = np.concatenate((inputs_chained),axis=0)
        outputs_chained = np.concatenate((targets_chained),axis=0) 
        
        return inputs_chained,outputs_chained
    
    def train(self,train_params,input_params, init_params=None):
    
        state_size,input_size,output_size,g,batch_size,num_iters,nstep_sizes,\
            init_step_size, rand_gen_start, random_key, decay_lr,\
            reg_type, reg_size, tasks_to_chain, save_dir, save_name = train_params
        
        # Allow continued training.
        if init_params is None:
            #Use first rand key to draw initial parameters
            init_params = self.create_rnn_params(input_size=input_size, 
                                            output_size=output_size,
                                            state_size=state_size,
                                            g=g,random_key=random_key)
            
    
        def training_loss(params, iter):
            inputs,targets = self.chain_task(input_params,tasks_to_chain)
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

class run_chaintasks(rnn_run):
    def __init__(self):
        super().__init__(self)
        self.tasks_to_chain = ['mem_pro','mem_anti']
        self.build_instance = build_chaintasks()
    
    def set_task_params(self,task_params):
        T, ntime, dt, bval, sval, zeros_beginning, input_size, output_size,\
            tasks_to_chain = task_params
        
        self.T = T
        self.ntime = ntime
        self.dt = dt
        self.bval = bval
        self.sval = sval
        self.zeros_beginning = zeros_beginning
        self.input_size = input_size
        self.output_size = output_size
        self.tasks_to_chain = tasks_to_chain
        
    def get_params(self):

        params_dict = dict(bval=self.bval,sval=self.sval,T=self.T,ntime=self.ntime,\
                        decay_learning_rate = self.decay_lr,\
                        input_size=self.input_size,output_size = self.output_size,\
                        state_size = self.state_size,g=self.g, batch_size=self.batch_size,\
                        num_iters=self.num_iters, nstep_sizes=self.nstep_sizes,\
                        init_step_size=self.init_step_size,run_number = self.run_number,\
                        save_dir=self.save_dir,save_name=self.save_name,\
                        rand_gen_start = self.rand_gen_start,\
                        tasks_to_chain = self.tasks_to_chain,\
                        random_key = self.random_key)
        
        rnn_input_params = self.random_key,self.bval, self.sval, self.T,\
            self.ntime, self.input_size, self.batch_size, self.zeros_beginning, self.save_name_task
            
        rnn_train_params = (self.state_size,self.input_size,self.output_size,\
                            self.g,self.batch_size,self.num_iters,self.nstep_sizes,\
                            self.init_step_size,self.rand_gen_start, self.random_key,\
                            self.decay_lr,self.reg_type,self.reg_size,self.tasks_to_chain,\
                                self.save_dir, self.save_name)
        return rnn_input_params, rnn_train_params, params_dict
    
        
    def run_training(self,params_to_tune,run_params):

        #Check inputs
        assert len(params_to_tune) == 8, 'Need 8 paramters in params_to_tune'
        assert len(run_params) == 2, 'Need 2 parameters in run_params'
        
        #Set params for current run
        self.set_params(params_to_tune,run_params)
        
        #Get params for training
        rnn_input_params, rnn_train_params, params_dict = self.get_params()
    
        #Check task setup 
        inputs,targets = self.build_instance.chain_tasks(input_params=rnn_input_params,
                                                         tasks_to_chain=self.tasks_to_chain)
            
        
        #Save using pickle (faster)
        filehandler = open(self.save_dir+"/RNN_init_params_"+self.save_name+'.pickle',"wb")
        pickle.dump(rnn_train_params,filehandler) 
        pickle.dump(rnn_input_params,filehandler)
        pickle.dump(params_dict,filehandler)
        filehandler.close()
        
        print('Run RNN Training. | Run Nr: %d'%(self.run_number))
        
        #Train RNN
        self.build_instance.train(train_params=rnn_train_params,input_params=rnn_input_params)