#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:27:02 2020

@author: cdmdc
"""
import jax.numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy import signal

#Build Context-Switch Task
def build_task(batch_size, input_params):
    ntrials = np.asarray(batch_size/2,dtype=int)
    bias_val, stddev_val, T, ntime = input_params
    dt = T/float(ntime)
    num_inputs = 2
    zeros_beginning = 10
    
    # biases_1xexw = np.expand_dims(bias_val * 2.0 * (numpy.random.rand(ntrials,nwninputs) -0.5), axis=0)
    stddev = stddev_val / np.sqrt(dt)
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
