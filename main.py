# The following code is a simple version of the distributed code at 
# https://github.com/Lham71/Working-Memory-Modeling 
#
# To run the program, navigate to the current code directory and 
# enter python main.py at the command line.
# 
# It will run a single experiment for the parameters listed below.
# 
# Simply change any of the parameters below if you want to run a 
# different experiment.
#
# The code requires that the following files are located in the same 
# location as main.py:
#     1. spm_task.py
#     2. train_force.py
#     3. posthoc_test.py
#     4. allDigCNNMNIST
#
# The code also requires a results folder to save the results
#
# Please contact steven.wilson@ou.edu if you have questions or need
# help with any of the code.

################# import packages ###################
import sys
import os
import pickle as pk
import numpy as np
from spm_task import * 
from train_force import *
from posthoc_tests import *

################# set parameters ###################
params = dict()                                               # dictionary to hold experiment parameters

net_params = dict()                                           # dictionary to hold network parameters
net_params['d_input'] = 2
net_params['d_output'] = 1
net_params['tau'] = 1                                         # network time constant ?
net_params['dt'] = 0.1                                        # euler integration time steps
net_params['g'] = 0.8                                         # strength of synaptic interaction ?
net_params['pg'] = 0.4
net_params['N'] = 1000                                        # number of neurons
net_params['fb_var'] = 40
net_params['input_var'] = 50
params['network'] = net_params

t_intervals = dict()                                          # dictionary of task time intervals
t_intervals['fixate_on'], t_intervals['fixate_off'] = 0, 0
t_intervals['cue_on'], t_intervals['cue_off'] = 0, 0
t_intervals['stim_on'], t_intervals['stim_off'] = 10, 5
t_intervals['delay_task'] = 0
t_intervals['response'] = 5

task_params = dict()                                           # dictionary to hold task parameters
task_params['time_intervals'] = t_intervals                    # assign dictionary of task time intervals from above
task_params['t_trial'] = sum(t_intervals.values()) + t_intervals['stim_on'] + t_intervals['stim_off']
task_params['output_encoding'] = 1, 0.5, 1.5                   # how 0, 1, 2 are encoded (trial outputs that network should learn)
task_params['keep_perms'] = [(0, 0), (0, 1), (1, 0), (1, 1)]   # tuple of trials to perform
task_params['n_digits'] = 9                                    # number of digits in MNIST dataset
params['task'] = task_params

train_params = dict()                                           # dictionary to hold training parameters
train_params['update_step'] = 2                                 # update steps of FORCE
train_params['alpha_w'] = 1.
train_params['alpha_d'] = 1.
train_params['n_train'] = 3500                                  # number of trials for training
train_params['n_train_ext'] = 0                                 # number of extra training trials
train_params['n_test'] = 20                                     # number of trials for testing
train_params['init_dist'] = 'Gauss'                             # distribution for weight initialization -- either 'Gauss' or 'Uniform'
train_params['activation'] = 'tanh'                             # activation function
train_params['FORCE'] = False
train_params['epsilon'] = [0.005, 0.01, 0.05, 0.1]
params['train'] = train_params


other_params = dict()                                            # dictionary to hold miscellaneous parameters

########### comment out for now and set name to static value ###########
# other_params['name'] = '_'.join(['{}'.format(val) if type(val) != list
#                                 else '{}'.format(''.join([str(s) for s in val])) for k, val in kwargs.items()])

other_params['name'] = 'simple-working-memory'
print('name is = ',other_params['name']  )
other_params['n_plot'] = 10
other_params['seed'] = 0  # random seed default is 0
params['msc'] = other_params

################# get all digit representations from vae ###################
def get_digits_reps():
    with open('allDigCNNMNIST', 'rb') as f:
        z_mean, z_log_var, z_sample = pk.load(f)
        x_test = pk.load(f)
        y_test = pk.load(f)

    y_test, x_test = np.array(y_test), x_test.reshape([x_test.shape[0], 28, 28])

    return y_test, z_sample

# labels are true MNIST labels 
# digits_rep are vae MNIST digit representations
labels, digits_rep = get_digits_reps()

################# get all digit representations from vae ###################
def get_digits_reps():
    with open('allDigCNNMNIST', 'rb') as f:
        z_mean, z_log_var, z_sample = pk.load(f)
        x_test = pk.load(f)
        y_test = pk.load(f)

    y_test, x_test = np.array(y_test), x_test.reshape([x_test.shape[0], 28, 28])

    return y_test, z_sample

# labels are true MNIST labels 
# digits_rep are vae MNIST digit representations
labels, digits_rep = get_digits_reps()

################# set pram representation ###################
task_prs = params['task']
train_prs = params['train']
net_prs = params['network']
msc_prs = params['msc']

################# sum task experiment ###################
task = sum_task_experiment(task_prs['n_digits'], train_prs['n_train'], train_prs['n_train_ext'], train_prs['n_test'], task_prs['time_intervals'],
                           net_prs['dt'], task_prs['output_encoding'], task_prs['keep_perms'] , digits_rep, labels, msc_prs['seed'])

exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()

print("training")
if not train_prs['FORCE']:
    print('FORCE Reinforce IS RUNNING\n')
    x_train, params = train(params, exp_mat, target_mat, dummy_mat, input_digits, dist=train_prs['init_dist'])
    
elif train_prs['FORCE']:
    print('FORCE IS RUNNING\n')
    x_train, params = train_FORCE(params, exp_mat, target_mat, dummy_mat, input_digits, dist=train_prs['init_dist'])

x_ICs, r_ICs, internal_x = test(params, x_train, exp_mat, target_mat, dummy_mat, input_digits)

error_ratio = error_rate(params, x_ICs, digits_rep, labels)

# save experiment results
save_data_variable_size(params, x_ICs, r_ICs, error_ratio, name=params['msc']['name'], prefix='train', dir='results/')   

################# post-hoc tests  ###################
print("post-hoc testing")
ph_params = set_posthoc_params(x_ICs, r_ICs)

trajectories, unique_z_mean, unique_zd_mean, attractor = attractor_type(params, ph_params, digits_rep, labels)

_, _, _, attractor_nozero = attractor_type_nozero(params, ph_params, digits_rep, labels)

all_F_norms = asymp_trial_end(params, ph_params)

all_delay_traj, all_z, all_zd = vf_delays(params, ph_params, digits_rep, labels)

save_data_variable_size(ph_params,  unique_z_mean, unique_zd_mean, attractor, attractor_nozero,
                        all_F_norms,  name=params['msc']['name'], prefix='fp_test', dir='results/')

sat_ratio_delay, sat_ratio_resp = saturation_percentage(params, ph_params, digits_rep, labels)

trials_failed, dsct_failed_ratio, intrp_failed_ratio = 0, 0 , 0

# save post-hoc test results
save_data_variable_size(trials_failed, dsct_failed_ratio, intrp_failed_ratio,
                        sat_ratio_delay, sat_ratio_resp, name=params['msc']['name'], prefix='ext_test', dir='results/')

print('DONE')