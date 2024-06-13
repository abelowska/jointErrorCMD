functions {
    /* Wiener diffusion log-PDF for a single response (adapted from brms 1.10.2)
    * Arguments:
    *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    *   boundary: boundary separation parameter > 0
    *   ndt: non-decision time parameter > 0
    *   bias: initial bias parameter in [0, 1]
    *   drift: drift rate parameter
    * Returns:
    *   a scalar to be added to the log posterior
    */
    real diffusion_lpdf(real Y, real boundary, real ndt, real bias, real drift) {
        if (Y >= 0) {
            return wiener_lpdf( abs(Y) | boundary, ndt, bias, drift );
        } else {
            return wiener_lpdf( abs(Y) | boundary, ndt, 1-bias, -drift );
        }
    }

    real participant_level_diffusion_lpdf(
        vector y, 
        real boundary,
        real boundary_cond,
        real boundary_ne,
        real boundary_pre_acc,
        real boundary_ne_pre_acc,
        real boundary_ne_cond,
        real boundary_pre_acc_cond,
        real boundary_ne_pre_acc_cond,
        real ndt, 
        real bias, 
        real drift, 
        real drift_cond,
        real drift_ne,
        real drift_pre_acc,
        real drift_ne_pre_acc, 
        real drift_ne_cond,
        real drift_pre_acc_cond,
        real drift_ne_pre_acc_cond,
        vector condition, 
        vector pre_ne,
        vector pre_acc,
        int n_trials
    ) {
        vector[n_trials] participant_level_likelihood;
        
        for (t in 1:n_trials) {
            if (abs(y[t]) - ndt > 0) {
                participant_level_likelihood[t] = diffusion_lpdf(y[t] | boundary  + boundary_cond*condition[t] + boundary_ne*pre_ne[t] + boundary_pre_acc*pre_acc[t] + boundary_ne_pre_acc*pre_ne[t]*pre_acc[t] + boundary_ne_cond*pre_ne[t]*condition[t] + boundary_pre_acc_cond*pre_acc[t]*condition[t] + boundary_ne_pre_acc_cond*pre_ne[t]*pre_acc[t]*condition[t], ndt, bias, drift + drift_cond*condition[t]);
            } else {
                participant_level_likelihood[t] = diffusion_lpdf(ndt | boundary  + boundary_cond*condition[t] + boundary_ne*pre_ne[t] + boundary_pre_acc*pre_acc[t] + boundary_ne_pre_acc*pre_ne[t]*pre_acc[t] + boundary_ne_cond*pre_ne[t]*condition[t] + boundary_pre_acc_cond*pre_acc[t]*condition[t] + boundary_ne_pre_acc_cond*pre_ne[t]*pre_acc[t]*condition[t], ndt, bias, drift + drift_cond*condition[t]);
            }
        }
        return(sum(participant_level_likelihood));
    }
    
    real log_likelihood_diffusion_lpdf(
        real y, 
        real boundary,
        real boundary_cond,
        real boundary_ne,
        real boundary_pre_acc,
        real boundary_ne_pre_acc,
        real boundary_ne_cond, 
        real boundary_pre_acc_cond,
        real boundary_ne_pre_acc_cond,
        real ndt, 
        real bias, 
        real drift, 
        real drift_cond,
        real drift_ne,
        real drift_pre_acc,
        real drift_ne_pre_acc, 
        real drift_ne_cond,
        real drift_pre_acc_cond,
        real drift_ne_pre_acc_cond,
        real condition, 
        real pre_ne,
        real pre_acc
    ) {
        if (abs(y) - ndt > 0) {
            return diffusion_lpdf(y | boundary  + boundary_cond*condition + boundary_ne*pre_ne + boundary_pre_acc*pre_acc + boundary_ne_pre_acc*pre_ne*pre_acc + boundary_ne_cond*pre_ne*condition + boundary_pre_acc_cond*pre_acc*condition + boundary_ne_pre_acc_cond*pre_ne*pre_acc*condition, ndt, bias, drift + drift_cond*condition + drift_ne*pre_ne + drift_ne_cond*pre_ne*condition + drift_pre_acc*pre_acc + drift_pre_acc_cond*pre_acc*condition + drift_ne_pre_acc*pre_ne*pre_acc + drift_ne_pre_acc_cond*pre_ne*pre_acc*condition);
        } else {
            return diffusion_lpdf(ndt | boundary  + boundary_cond*condition + boundary_ne*pre_ne + boundary_pre_acc*pre_acc + boundary_ne_pre_acc*pre_ne*pre_acc + boundary_ne_cond*pre_ne*condition + boundary_pre_acc_cond*pre_acc*condition + boundary_ne_pre_acc_cond*pre_ne*pre_acc*condition, ndt, bias, drift + drift_cond*condition + drift_ne*pre_ne + drift_ne_cond*pre_ne*condition + drift_pre_acc*pre_acc + drift_pre_acc_cond*pre_acc*condition + drift_ne_pre_acc*pre_ne*pre_acc + drift_ne_pre_acc_cond*pre_ne*pre_acc*condition);
        }
    }
}

data {
    int<lower=1> N; // Number of trial-level observations
    int<lower=1> n_conditions; // Number of conditions (congruent and incongruent)
    int<lower=1> n_participants; // Number of participants

    array[n_participants, 2] int participants_trials_slices; // slices TODO
    vector[N] y; // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    vector[N] condition; // Contrast coded condition: -1 for erroneous and 1 for correct response respectively
    vector[N] pre_acc; // Contrast coded accuracy on previous trial
    vector[N] pre_ne; // centered correct/error negativity on previous trial
    array[N] int<lower=1> participant; // Participant index
}

parameters {
    vector<lower=0, upper=0.3>[n_participants] participants_ter;
    vector<lower=0, upper=3>[n_participants] participants_alpha;
    vector[n_participants] participants_alpha_cond;
    vector[n_participants] participants_alpha_ne;
    vector[n_participants] participants_delta;
    vector[n_participants] participants_delta_cond;
    vector[n_participants] participants_delta_ne;
        
    real<lower=0> ter;
    real<lower=0, upper=3> alpha;
    real alpha_cond;
    real alpha_ne;
    real delta;
    real delta_cond;
    real delta_ne;
    
    real<lower=0> ter_sd; 
    real<lower=0> alpha_sd;
    real<lower=0> alpha_cond_sd;
    real<lower=0> alpha_ne_sd;
    real<lower=0> delta_sd; 
    real<lower=0> delta_cond_sd;
    real<lower=0> delta_ne_sd;
    
    
    // Population-level effects
    real alpha_pre_acc; 
    real alpha_ne_pre_acc; 
    real alpha_ne_cond;
    real alpha_pre_acc_cond; 
    real alpha_ne_pre_acc_cond; 
    
    real delta_pre_acc; 
    real delta_ne_pre_acc;
    real delta_ne_cond;
    real delta_pre_acc_cond;
    real delta_ne_pre_acc_cond;
}

model {

    // ##########
    // Between-participant variability priors
    // ##########
    ter_sd ~ gamma(.3,1);
    alpha_sd ~ gamma(1,1);
    alpha_cond_sd ~ gamma(1,1);
    alpha_ne_sd ~ gamma(1,1);
    delta_sd ~ gamma(1,1);
    delta_cond_sd ~ gamma(1,1);
    delta_ne_sd ~ gamma(1,1);

    // ##########
    // Hierarchical parameters priors
    // ##########
    ter ~ normal(.1, .2);
    alpha ~ normal(1, 1) T[0, 3];
    alpha_cond ~ normal(0, 1);  // 0.2
    alpha_ne ~ normal(0, 0.5);
    delta ~ normal(0, 2);
    delta_cond ~ normal(0, 2);
    delta_ne ~ normal(0,1);

    // ##########
    // Population-level boundary effects
    // ##########
    alpha_pre_acc ~ normal(0,0.2);  
    alpha_ne_pre_acc ~ normal(0, 0.1);
    alpha_ne_cond ~ normal(0, 0.1);
    alpha_pre_acc_cond ~ normal(0,0.2);  
    alpha_ne_pre_acc_cond ~ normal(0, 0.1);
    
    // ##########
    // Population-level drift effects
    // ##########
    delta_ne_pre_acc ~ normal(0, 1);
    delta_pre_acc ~ normal(0,1);  
    delta_ne_cond ~ normal(0, 1);
    delta_pre_acc_cond ~ normal(0,1);
    delta_ne_pre_acc_cond ~ normal(0, 1);


    // ##########
    // Participant-level DDM parameter priors
    // ##########
    for (p in 1:n_participants) {

        // Participant-level non-decision time
        participants_ter[p] ~ normal(ter, ter_sd) T[0, .3];

        // Participant-level boundary parameter (speed-accuracy tradeoff)
        participants_alpha[p] ~ normal(alpha, alpha_sd) T[0, 3];
        participants_alpha_cond[p] ~ normal(alpha_cond, alpha_cond_sd);
        participants_alpha_ne[p] ~ normal(alpha_ne, alpha_ne_sd);

        //Participant-level drift rate
        participants_delta[p] ~ normal(delta, delta_sd);
        participants_delta_cond[p] ~ normal(delta_cond, delta_cond_sd); 
        participants_delta_ne[p] ~ normal(delta_ne, delta_ne_sd);  
                
        target += participant_level_diffusion_lpdf( y[participants_trials_slices[p][1]:participants_trials_slices[p][2]] | participants_alpha[p], participants_alpha_cond[p], participants_alpha_ne[p], alpha_pre_acc, alpha_ne_pre_acc, alpha_ne_cond, alpha_pre_acc_cond, alpha_ne_pre_acc_cond, participants_ter[p], 0.5, participants_delta[p], participants_delta_cond[p], participants_delta_ne[p], delta_pre_acc, delta_ne_pre_acc, delta_ne_cond, delta_pre_acc_cond, delta_ne_pre_acc_cond, condition[participants_trials_slices[p][1]:participants_trials_slices[p][2]], pre_ne[participants_trials_slices[p][1]:participants_trials_slices[p][2]],pre_acc[participants_trials_slices[p][1]:participants_trials_slices[p][2]], (participants_trials_slices[p][2] - participants_trials_slices[p][1] + 1));         
    }
}
generated quantities { 
   vector[N] log_lik;     
    
   // Wiener likelihood
    for (i in 1:N) {
        // Log density for DDM process
         log_lik[i] = log_likelihood_diffusion_lpdf(y[i] | participants_alpha[participant[i]], participants_alpha_cond[participant[i]], participants_alpha_ne[participant[i]], alpha_pre_acc, alpha_ne_pre_acc, alpha_ne_cond, alpha_pre_acc_cond, alpha_ne_pre_acc_cond, participants_ter[participant[i]], 0.5, participants_delta[participant[i]], participants_delta_cond[participant[i]], participants_delta_ne[participant[i]], delta_pre_acc, delta_ne_pre_acc, delta_ne_cond, delta_pre_acc_cond, delta_ne_pre_acc_cond, condition[i], pre_ne[i], pre_acc[i]);
    }
}
