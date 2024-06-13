'''
HDDM models informed with EEG and pre-trial accuracy 
'''
# Imports
from cmdstanpy import CmdStanModel
import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
import time
from contextlib import redirect_stdout
import sys
import yaml

# Read script config
yaml_file = sys.argv[1]
with open(yaml_file, "r") as file:
    config = yaml.safe_load(file)

print(config)

###############################################
# Model compiling
###############################################

# Define model to fit
type = config['model'] 
model_name = f'wiener_{type}_model.stan'
print(f'Processing model: {model_name}')

# Compile the model
stan_file = os.path.join("./models/", model_name)
print(stan_file)
hddm_model = CmdStanModel(
    stan_file=stan_file, 
    cpp_options={'STAN_THREADS': True}, 
    force_compile=True
)

###############################################
# Read data
###############################################

# Define data file
data_file = os.path.join('./data/', config['data'])

# Read data to dataframe
with open(data_file, 'r') as file:
    data = json.load(file)

data_df = pd.DataFrame(
    {
        'participant_index': data['participant'],
         'rt': abs(np.array(data['y']))
    }
)

###############################################
# Model fitting
###############################################

# Define fitting parameters
random_state = 42

num_chains = config['fit_params']['n_chains']
warmup = config['fit_params']['warmup']
num_samples = config['fit_params']['num_samples']
thin=config['fit_params']['thin']
threads_per_chain = config['fit_params']['threads_per_chain']

# Define initial values
n_participants = data['n_participants']
min_rt = np.zeros(n_participants)
for idx, participant_idx in enumerate(np.unique(data['participant'])):
    participant_rts = data_df[data_df['participant_index'] == participant_idx]['rt'].to_numpy()
    min_rt[idx] = np.min(abs(participant_rts))

try:
    initial_path = config['initial_path']
    initials = np.load(initial_path, allow_pickle=True).tolist()

    assert len(initials) == num_chains

except:
    print('Calculating initial values')
    initials = []
    for c in range(0, num_chains):
        chain_init = {               
            'ter_sd': np.random.uniform(.01, .2),
            'alpha_sd': np.random.uniform(.01, 1.),
            'alpha_cond_sd': np.random.uniform(.01, 1.), # <- was 0.5
            'delta_sd': np.random.uniform(.1, 3.),
            'delta_cond_sd': np.random.uniform(.1, 3.),
            
            'alpha_ne_sd': np.random.uniform(.01, .2), # <- works quite nice with .01, .2, works with .01, 1 
            'delta_ne_sd': np.random.uniform(.01, 1), # 0.2
    
            'ter': np.random.uniform(0.05, .3),
            'alpha': np.random.uniform(1, 2), #0.2 ## <- does not work with < 1
            'alpha_cond': np.random.uniform(-.5, .5), # <- was -.1, .1 and works a little bit better
            'delta': np.random.uniform(-4., 4.),
            'delta_cond': np.random.uniform(-4., 4.),
    
            'alpha_ne': np.random.uniform(-.05, .05), # <- does not work with -0.1, 0.1
            'alpha_pre_acc': np.random.uniform(-0.1, .1), 
            'alpha_ne_pre_acc': np.random.uniform(-.05, .05), # does not work with -0.1, 0.1
            'alpha_ne_cond': np.random.uniform(-.05, .05), # <- does not work with -0.1, 0.1
            'alpha_pre_acc_cond': np.random.uniform(-0.1, .1), 
            'alpha_ne_pre_acc_cond': np.random.uniform(-.05, .05), # does not work with -0.1, 0.1
    
            'delta_ne': np.random.uniform(-.5, .5),
            'delta_pre_acc': np.random.uniform(-.5, .5),
            'delta_ne_pre_acc': np.random.uniform(-.5, .5),
            'delta_ne_cond': np.random.uniform(-.5, .5),
            'delta_pre_acc_cond': np.random.uniform(-.5, .5),
            'delta_ne_pre_acc_cond': np.random.uniform(-.5, .5),
            
            'participants_ter': np.random.uniform(0.05, .3, size=n_participants),
            'participants_alpha': np.random.uniform(1, 2., size=n_participants), ## <- does not work with <1
            'participants_alpha_cond': np.random.uniform(-0.5, .5, size=n_participants), # <- was -.1, .1 and works a little bit better 
            'participants_delta': np.random.uniform(-4., 4., size=n_participants),
            'participants_delta_cond': np.random.uniform(-4., 4., size=n_participants),
            
            'participants_alpha_ne': np.random.uniform(-.05, .05, size=n_participants),
            'participants_delta_ne': np.random.uniform(-.5, .5, size=n_participants),
        }
    
        for p in range(0, n_participants):
            chain_init['participants_ter'][p] = np.random.uniform(0., min_rt[p]/2)
    
        initials.append(chain_init)

# Perform fitting
with open('stan_logs.txt', 'a') as f:
    with redirect_stdout(f):
        start = time.time()
        fit = hddm_model.sample(
            data=data_file,
            chains=num_chains, 
            seed=random_state, 
            inits=initials, 
            iter_warmup=warmup, 
            iter_sampling=num_samples,
            parallel_chains=num_chains,
            threads_per_chain= 4,
            show_progress=True,
            show_console=True,
        )
        end = time.time()

print(f'Fitting took: {end - start}')

# Save MCMC fit object
results_path = f'./results/{type}/'
try:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    fit.save_csvfiles(dir=results_path)
except Exception as e:
    print(e)