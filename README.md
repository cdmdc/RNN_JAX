# RNN_JAX
Training recurrent neural networks (RNNs) based on Jax. 

########################
rnn_jax_build: functions to build RNN

rnn_jax_run: functions to run RNN

Dependencies: jax=0.1.72, numpy=1.19.2, matplotlib=3.3.2, pickle5


Currently training on Context Switch Task. However, by overriding build_task in rnn_jax_build one can train on any task. 

run_rnn class in rnn_jax_run is initialised with parameters that work well for training neural tasks, such as the context switch task, and achieve good similarity to brain responses. These parameter settings can be overridden, however, for individual runs or for hyper parameter tuning using set_params for a given run.

Task related variables can be set using set_task_params after initialising run_rnn class in rnn_jax_run.

#######################
rnn_jax_build_chaintasks: inherits from rnn_jax_build and extends functionality by adding a new build task function that allows training on nine different neuroscience tasks (see Duncker et al NeurIPS 2020 for task descriptions), and by including a function (chain_tasks) that allows chaining tasks sequentially.

rnn_jax_run_chaintasks: train RNN on multiple tasks sequentially.

Currently set up to run training on 'mem_pro' and 'mem_anti' tasks sequentially. Will be extended to train on all nine tasks sequentially in the future. 