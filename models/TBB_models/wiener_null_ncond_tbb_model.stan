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

    real partial_diffusion_lpdf(
        array[] real y_slice,
        int start,
        int end,
        vector boundary,
        vector ndt, 
        real bias, 
        vector drift, 
        array[] int participant
    ) {
        int n_trials = end - start + 1;
        int global_index = start;
        vector[n_trials] partial_sum_level_likelihood;
        
        for (t in 1:n_trials) {
            if (abs(y_slice[t]) - ndt[participant[global_index]] > 0) {
                partial_sum_level_likelihood[t] = diffusion_lpdf(y_slice[t] | boundary[participant[global_index]], ndt[participant[global_index]], bias, drift[participant[global_index]]);
            } else {
                partial_sum_level_likelihood[t] = diffusion_lpdf(ndt[participant[global_index]] | boundary[participant[global_index]], ndt[participant[global_index]], bias, drift[participant[global_index]]);
            }
            global_index = global_index + 1;
        }
        return(sum(partial_sum_level_likelihood));
    }
    
    real log_likelihood_diffusion_lpdf(
        real y, 
        real boundary,
        real ndt, 
        real bias, 
        real drift 
    ) {
        if (abs(y) - ndt > 0) {
            return diffusion_lpdf(y | boundary, ndt, bias, drift);
        } else {
            return diffusion_lpdf(ndt | boundary, ndt, bias, drift);
        }
    }
}

data {
    int<lower=1> N; // Number of trial-level observations
    int<lower=1> n_conditions; // Number of conditions (congruent and incongruent)
    int<lower=1> n_participants; // Number of participants

    array[n_participants, 2] int participants_trials_slices; // slices TODO
    array[N] real y; // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    vector[N] condition; // Contrast coded condition: -1 for erroneous and 1 for correct response respectively
    array[N] int<lower=1> participant; // Participant index
}

parameters {
    vector<lower=0, upper=0.3>[n_participants] participants_ter; 
    vector<lower=0, upper=3>[n_participants] participants_alpha; 
    vector[n_participants] participants_delta;
        
    real<lower=0> ter;
    real<lower=0, upper=3> alpha;
    real delta;
    
    real<lower=0> ter_sd;
    real<lower=0> alpha_sd;
    real<lower=0> delta_sd;
}

model {

    // ##########
    // Between-participant variability priors
    // ##########
    ter_sd ~ gamma(.3,1);
    alpha_sd ~ gamma(1,1);
    delta_sd ~ gamma(1,1);

    // ##########
    // Hierarchical parameters priors
    // ##########
    ter ~ normal(.1, .2);
    alpha ~ normal(1, 1) T[0, 3];
    delta ~ normal(0, 2);

    // ##########
    // Participant-level DDM parameter priors
    // ##########
    for (p in 1:n_participants) {

        // Participant-level non-decision time
        participants_ter[p] ~ normal(ter, ter_sd) T[0, .3];

        // Participant-level boundary parameter (speed-accuracy tradeoff)
        participants_alpha[p] ~ normal(alpha, alpha_sd) T[0, 3];
                
        //Participant-level drift rate
        participants_delta[p] ~ normal(delta, delta_sd);
    }

    int grainsize = 100;     
    target += reduce_sum(partial_diffusion_lpdf, y, grainsize, participants_alpha, participants_ter, 0.5, participants_delta, participant);         
}

generated quantities { 
   vector[N] log_lik;     
    
   // Wiener likelihood
    for (i in 1:N) {
        // Log density for DDM process
         log_lik[i] = log_likelihood_diffusion_lpdf(y[i] | participants_alpha[participant[i]], participants_ter[participant[i]], 0.5, participants_delta[participant[i]]);
    }
}
