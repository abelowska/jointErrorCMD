import numpy as np
import pandas as pd
from datetime import datetime
import cmdstanpy
from scipy import stats
from scipy.stats import halfnorm, norm
import arviz.labels as azl
import arviz as az
import math
import re

def increment_index(idx):
    return re.sub(r'\[(\d+)\]', lambda x: f'[{int(x.group(1)) + 1}]', idx)
        
def calculate_bayes_factor_participant_level(
        fit_df, 
        parameters_list, 
        prior_distribution, 
        hierarchical=False,
        log=False
):
    prior_kde = calculate_hierarchical_prior_kde(prior_distribution, N=100000)
    participants_bf = []

    for participants_parameter in parameters_list:
        participant_bf = calculate_bayes_factor(fit_df, participants_parameter, prior_kde, log=log)
        participants_bf.append(participant_bf)

    return np.array(participants_bf).flatten()

def calculate_hierarchical_prior_kde(hyper_prior_distributions, N = 100000):
    if isinstance(hyper_prior_distributions, list):
        prior_samples = []
        for hyper_prior in hyper_prior_distributions:
            mean_hyper_prior = hyper_prior['mean']
            type, sd_hyper_prior = hyper_prior['sd']

            mean_samples = np.random.normal(loc=mean_hyper_prior['loc'], scale=mean_hyper_prior['scale'], size=(N,))
            if type == 'gamma':
                sd_samples = np.random.gamma(shape = sd_hyper_prior['shape'], scale=sd_hyper_prior['scale'], size=(N,))
            else:
                 sd_samples = halfnorm.rvs(loc = sd_hyper_prior['shape'], scale = sd_hyper_prior['scale'], size=(N,))      

            for i in range(0,N):
                prior_sample = np.random.normal(loc=mean_samples[i], scale=sd_samples[i], size=None)
                prior_samples.append(prior_sample)
        prior_samples = np.array(prior_samples)
                
    else:
        mean_hyper_prior = hyper_prior_distributions['mean']
        type, sd_hyper_prior = hyper_prior_distributions['sd']
        
        mean_samples = np.random.normal(loc=mean_hyper_prior['loc'], scale=mean_hyper_prior['scale'], size=(N,))
        if type == 'gamma':
            sd_samples = np.random.gamma(shape = sd_hyper_prior['shape'], scale=sd_hyper_prior['scale'], size=(N,))
        else:
            sd_samples = halfnorm.rvs(loc = sd_hyper_prior['shape'], scale = sd_hyper_prior['scale'], size=(N,))
        
        prior_samples = []
        for i in range(0,N):
            prior_sample = np.random.normal(loc=mean_samples[i], scale=sd_samples[i], size=None)
            prior_samples.append(prior_sample)
        prior_samples = np.array(prior_samples)
        
    # Prior density of hierarchical effect parameters
    prior_density = stats.gaussian_kde(prior_samples)
    
    return prior_density

def calculate_bayes_factor(fit_df, parameter, prior_kde, log=False):
    parameter_samples = fit_df[parameter].to_numpy()

    # Estimate density curves from samples
    parameter_kde = stats.gaussian_kde(parameter_samples)
    if log:
        parameter_pdf = parameter_kde.logpdf
        prior_pdf = prior_kde.logpdf
    else:
        parameter_pdf = parameter_kde
        prior_pdf = prior_kde.pdf
        
    # Calculate Bayes Factors 10, evidence against the null hypothesis
    bayes_factor_10 = prior_pdf(0) / parameter_pdf(0)
        
    return bayes_factor_10

def get_summary_with_bayes_factor(fit, inference_data, priors_dict, variables_to_track, log=False, percentiles=(2, 98)):
    fit_df = fit.draws_pd()
    bayes_factors = dict()
    bayes_factors_log = dict()
    
    for parameter_name in priors_dict.keys():
        prior_dist, is_hierarchical = priors_dict[parameter_name]
    
        if is_hierarchical:
            parameters_list = [variable for variable in variables_to_track if parameter_name in variable ]
            
            participants_bf = calculate_bayes_factor_participant_level(
                fit_df,
                parameters_list=parameters_list,
                prior_distribution = prior_dist,
                hierarchical=True
            )
            bayes_factors_hierarchical = dict(zip(parameters_list, participants_bf))
            bayes_factors.update(bayes_factors_hierarchical)
        else:
            if isinstance(prior_dist, list):
                samples = np.array([])
                for dist in prior_dist:
                    current_samples = dist.rvs(size=100000)
                    samples = np.concatenate([samples, current_samples])
                prior_kde = stats.gaussian_kde(samples)
            else:
                 prior_kde = stats.gaussian_kde(prior_dist.rvs(size=100000))
                
            bf = calculate_bayes_factor(
                fit_df, 
                parameter=parameter_name,
                prior_kde = prior_kde,
            )
            bayes_factors_population = dict(zip([parameter_name], bf))
            bayes_factors.update(bayes_factors_population)
    if log:
        for parameter_name in priors_dict.keys():
            prior_dist, is_hierarchical = priors_dict[parameter_name]
        
            if is_hierarchical:
                parameters_list = [variable for variable in variables_to_track if parameter_name in variable ]
                
                participants_bf = calculate_bayes_factor_participant_level(
                    fit_df,
                    parameters_list=parameters_list,
                    prior_distribution = prior_dist,
                    hierarchical=True,
                    log=True
                )
                bayes_factors_hierarchical = dict(zip(parameters_list, participants_bf))
                bayes_factors_log.update(bayes_factors_hierarchical)
            else:
                if isinstance(prior_dist, list):
                    samples = np.array([])
                    for dist in prior_dist:
                        current_samples = dist.rvs(size=100000)
                        samples = np.concatenate([samples, current_samples])
                    prior_kde = stats.gaussian_kde(samples)
                else:
                    prior_kde = stats.gaussian_kde(prior_dist.rvs(size=100000))
                        
                bf = calculate_bayes_factor(
                    fit_df, 
                    parameter=parameter_name,
                    prior_kde = prior_kde,
                    log=True
                )
                bayes_factors_population = dict(zip([parameter_name], bf))
                bayes_factors_log.update(bayes_factors_population)
            
        
    summary_df = fit.summary(percentiles=(2, 98))
    summary_df_az = az.summary(inference_data, hdi_prob=0.95, round_to="none")
    summary_df_az.index = summary_df_az.index.map(increment_index)
    
    bayes_factors_df = pd.DataFrame.from_dict(bayes_factors, orient='index', columns=['Bayes_factor'])
    bayes_factors_log_df = pd.DataFrame.from_dict(bayes_factors_log, orient='index', columns=['Bayes_factor_log'])
    result_df = pd.concat([summary_df, bayes_factors_df, bayes_factors_log_df], axis=1)
    result_az_df = pd.concat([summary_df_az, bayes_factors_df, bayes_factors_log_df], axis=1)

    return result_df, result_az_df

def waic(fit_df, col='log_lik'):
    """Calculates the Watanabe-Akaike information criteria.
    Calculates pWAIC1 and pWAIC2
    according to http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf
    Parameters
    ----------
   
    Returns
    -------
    model_statistics_df: pd.DataFrame
        FataFrame containing lppd (log pointwise predictive density),
        p_waic, waic, waic_se (standard error of the waic), and
        pointwise_waic (when `pointwise` is True).
    """
    log_likelihood = fit_df.filter(like=col).to_numpy()

    N = log_likelihood.shape[1]
    likelihood = np.exp(log_likelihood)

    mean_l = np.mean(likelihood, axis=0) # N observations

    pointwise_lppd = np.log(mean_l)
    lppd = np.sum(pointwise_lppd)

    pointwise_var_l = np.var(log_likelihood, axis=0) # N observations
    var_l = np.sum(pointwise_var_l)

    pointwise_waic = - 2*pointwise_lppd +  2*pointwise_var_l
    waic = -2*lppd + 2*var_l
    waic_se = np.sqrt(N * np.var(pointwise_waic))

    model_statistics = {'lppd':lppd,
           'p_waic':var_l,
           'waic':waic,
           'waic_se':waic_se}

    model_statistics_df = pd.DataFrame.from_dict(model_statistics, orient='index', columns=['value'])
    
    return model_statistics_df

def format_number(x, limit=(1000, 0.01)):
    """Format numbers with scientific notation for large values or fixed point for small values."""
    if pd.isna(x): 
        return 'n.a'
    elif isinstance(x, (int, float)):
        if (abs(x) > limit[0]) or (abs(x) < limit[1]):
            return f"{x:.2e}" 
        else:
            return f"{x:.2f}" 
    return x
    
def get_fixed_effects_summary(
    results_df, 
    row_order, 
    mapping=None, 
    type='all',
    decimal_notation_limit=(1000, 0.01)
):
    # Drop unnecessary columns and make a copy of the DataFrame
    columns_to_drop = ['MCSE', 'N_Eff/s']
    filtered_df = results_df.drop(columns=columns_to_drop).copy()  # Ensure a copy is made
    
    # Filter for fixed effects
    fixed_effects = filtered_df[
        ~filtered_df['parameter'].str.contains(
            'participants|log|lp__|ter', 
            case=False, 
            na=False
        )
    ].copy()  # Ensure a copy is made

    format_func = lambda x: format_number(x, decimal_notation_limit)
    fixed_effects.loc[:, ['Mean', 'StdDev', 'Bayes_factor']] = fixed_effects[['Mean', 'StdDev', 'Bayes_factor']].map(format_func)
    
    # Format CrI values
    if '2%' in fixed_effects.columns and '98%' in fixed_effects.columns:
        fixed_effects['CrI'] = fixed_effects.apply(
            lambda row: f"[{format_number(row['2%'], limit=decimal_notation_limit)}, {format_number(row['98%'], limit=decimal_notation_limit)}]", axis=1
        )
    
    # Define type and name columns
    if type == 'all':
        fixed_effects['type'] = 'all'
        fixed_effects['name'] = fixed_effects['parameter']
    else:
        # Extract the prefix (before underscore) as 'type'. If there's no underscore, take the whole string as 'type'
        fixed_effects['type'] = fixed_effects['parameter'].str.extract(r'^([^_]+)_', expand=False).fillna(fixed_effects['parameter'])
        
        # Set 'name' by removing the prefix (if it exists); if no prefix, name will be the same as parameter
        fixed_effects['name'] = fixed_effects['parameter'].str.replace(r'^[^_]+_', '', regex=True)
        
        # Replace 'name' with 'intercept' if the 'type' and 'name' are the same or if the parameter has no underscore
        fixed_effects['name'] = np.where(
            (fixed_effects['type'] == fixed_effects['name']) | (fixed_effects['parameter'] == fixed_effects['type']),
            'intercept',
            fixed_effects['name']
        )
    
    # Drop unused columns
    fixed_effects = fixed_effects.drop(columns=['parameter', '2%', '98%', 'N_Eff', 'R_hat'])
    
    # Reorder columns
    column_order = ['Mean', 'StdDev', 'CrI', 'Bayes_factor', 'type', 'name']
    fixed_effects = fixed_effects[column_order]
    
    # Rename columns
    default_mapping = {'Mean': 'M', 'StdDev': 'SD', 'Bayes_factor': 'BF'}
    fixed_effects = fixed_effects.rename(columns=mapping if mapping else default_mapping)
    
    # Filter by type
    if type != 'all':
        fixed_effects = fixed_effects[fixed_effects['type'] == type]
    
    # Reindex by row_order
    fixed_effects = fixed_effects.set_index('name').reindex(row_order).reset_index()

    return fixed_effects

def get_fixed_effects_summary_az(
    results_df, 
    row_order, 
    mapping=None, 
    type='all',
    decimal_notation_limit=(1000, 0.01)
):
    # Drop unnecessary columns and make a copy of the DataFrame
    columns_to_drop = ['mcse_mean', 'mcse_sd']
    filtered_df = results_df.drop(columns=columns_to_drop).copy()  # Ensure a copy is made
    
    # Filter for fixed effects
    fixed_effects = filtered_df[
        ~filtered_df['parameter'].str.contains(
            'participants|log|lp__|ter', 
            case=False, 
            na=False
        )
    ].copy()  # Ensure a copy is made

    format_func = lambda x: format_number(x, decimal_notation_limit)
    fixed_effects.loc[:, ['mean', 'sd', 'Bayes_factor']] = fixed_effects[['mean', 'sd', 'Bayes_factor']].map(format_func)
    
    # Format HDI values
    if 'hdi_2.5%' in results_df.columns and 'hdi_97.5%' in results_df.columns:
        fixed_effects['HDI'] = fixed_effects.apply(
            lambda row: f"[{format_number(row['hdi_2.5%'], limit=decimal_notation_limit)}, {format_number(row['hdi_97.5%'], limit=decimal_notation_limit)}]", axis=1
        )
    
    # Define type and name columns
    if type == 'all':
        fixed_effects['type'] = 'all'
        fixed_effects['name'] = fixed_effects['parameter']
    else:
        # Extract the prefix (before underscore) as 'type'. If there's no underscore, take the whole string as 'type'
        fixed_effects['type'] = fixed_effects['parameter'].str.extract(r'^([^_]+)_', expand=False).fillna(fixed_effects['parameter'])
        
        # Set 'name' by removing the prefix (if it exists); if no prefix, name will be the same as parameter
        fixed_effects['name'] = fixed_effects['parameter'].str.replace(r'^[^_]+_', '', regex=True)
        
        # Replace 'name' with 'intercept' if the 'type' and 'name' are the same or if the parameter has no underscore
        fixed_effects['name'] = np.where(
            (fixed_effects['type'] == fixed_effects['name']) | (fixed_effects['parameter'] == fixed_effects['type']),
            'intercept',
            fixed_effects['name']
        )
    
    # Drop unused columns
    fixed_effects = fixed_effects.drop(columns=['parameter', 'hdi_2.5%', 'hdi_97.5%', 'ess_bulk', 'ess_tail', 'r_hat'])
    
    # Reorder columns
    column_order = ['mean', 'sd', 'HDI', 'Bayes_factor', 'type', 'name']
    fixed_effects = fixed_effects[column_order]
    
    # Rename columns
    default_mapping = {'mean': 'M', 'sd': 'SD', 'Bayes_factor': 'BF'}
    fixed_effects = fixed_effects.rename(columns=mapping if mapping else default_mapping)
    
    # Filter by type
    if type != 'all':
        fixed_effects = fixed_effects[fixed_effects['type'] == type]
    
    # Reindex by row_order
    fixed_effects = fixed_effects.set_index('name').reindex(row_order).reset_index()

    return fixed_effects