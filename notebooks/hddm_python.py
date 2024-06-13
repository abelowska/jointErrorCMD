from cmdstanpy import cmdstan_path, set_cmdstan_path, CmdStanModel
import os
import numpy as np
set_cmdstan_path('/stan/math_HOW-TO-USE/cmdstan-ddm-7pm')
cmdstan_path()
import pandas as pd

os.environ['STAN_NUM_THREADS'] = "12"

stan_file = os.path.join('../models/', 'wiener_simple_boundary.stan')
ndt_full_model = CmdStanModel(stan_file=stan_file, cpp_options={'STAN_THREADS': True})

data_file = os.path.join('../data/', 'stan_data_cleaned.data.json')

df = pd.read_csv('../data/twentythree_participants_post_eeg_test_set_with_sequence_global_rt_thresholds.csv').drop(columns='Unnamed: 0')
df_no_nans = df.dropna()
