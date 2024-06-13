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
        vector boundary_ne,
        real boundary_pre_acc,
        real boundary_ne_pre_acc,
        vector ndt, 
        real bias, 
        vector drift, 
        vector pre_ne,
        vector pre_acc,
        array[] int participant
    ) {
        int n_trials = end - start + 1;
        int global_index = start;
        vector[n_trials] partial_sum_level_likelihood;
        
        for (t in 1:n_trials) {
            if (abs(y_slice[t]) - ndt[participant[global_index]] > 0) {
                partial_sum_level_likelihood[t] = diffusion_lpdf(y_slice[t] | boundary[participant[global_index]] + boundary_ne[participant[global_index]]*pre_ne[global_index] + boundary_pre_acc*pre_acc[global_index] + boundary_ne_pre_acc*pre_ne[global_index]*pre_acc[global_index], ndt[participant[global_index]], bias, drift[participant[global_index]]);
            } else {
                partial_sum_level_likelihood[t] = diffusion_lpdf(ndt[participant[global_index]] | boundary[participant[global_index]] + boundary_ne[participant[global_index]]*pre_ne[global_index] + boundary_pre_acc*pre_acc[global_index] + boundary_ne_pre_acc*pre_ne[global_index]*pre_acc[global_index], ndt[participant[global_index]], bias, drift[participant[global_index]]);
            }
            global_index = global_index + 1;
        }
        return(sum(partial_sum_level_likelihood));
    }
    
    real log_likelihood_diffusion_lpdf(
        real y, 
        real boundary,
        real boundary_ne,
        real boundary_pre_acc,
        real boundary_ne_pre_acc,
        real ndt, 
        real bias, 
        real drift, 
        real pre_ne,
        real pre_acc
    ) {
        if (abs(y) - ndt > 0) {
            return diffusion_lpdf(y | boundary + boundary_ne*pre_ne + boundary_pre_acc*pre_acc + boundary_ne_pre_acc*pre_ne*pre_acc, ndt, bias, drift);
        } else {
            return diffusion_lpdf(ndt | boundary + boundary_ne*pre_ne + boundary_pre_acc*pre_acc + boundary_ne_pre_acc*pre_ne*pre_acc, ndt, bias, drift);
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
    vector[N] pre_acc; // Contrast coded accuracy on previous trial
    vector[N] pre_ne; // centered correct/error negativity on previous trial
    array[N] int<lower=1> participant; // Participant index
}

parameters {
    vector<lower=0, upper=0.3>[n_participants] participants_ter; // Participant-level Non-decision time
    vector<lower=0, upper=3>[n_participants] participants_alpha; // Participant-level Boundary parameter (speed-accuracy tradeoff)
    vector[n_participants] participants_alpha_ne;  
    vector[n_participants] participants_delta; // Participant-level drift-rate
        
    real<lower=0> ter; // Hierarchical non-decision time
    real<lower=0, upper=3> alpha; // Hierarchical boundary parameter (speed-accuracy tradeoff)
    real alpha_ne;
    real delta; // Hierarchical drift-rate
    
    real<lower=0> ter_sd; // Between-participants variability in non-decision time
    real<lower=0> alpha_sd; // Between-participants variability in boundary parameter (speed-accuracy tradeoff)
    real<lower=0> alpha_ne_sd;
    real<lower=0> delta_sd; // Between-participants variability in drift-rate    
    
    // Non Hierarchical
    real alpha_pre_acc; 
    real alpha_ne_pre_acc; 
}

model {

    // ##########
    // Between-participant variability priors
    // ##########
    ter_sd ~ gamma(.3,1);
    alpha_sd ~ gamma(1,1);
    alpha_ne_sd ~ gamma(1,1); // works quite nice with 0.3, 1 and really nice with (1,1)
    delta_sd ~ gamma(1,1);

    // ##########
    // Hierarchical parameters priors
    // ##########
    ter ~ normal(.1, .2);
    alpha ~ normal(1, 1) T[0, 3];
    alpha_ne ~ normal(0, 0.5); // was 0.5 and works super, with 1 worser
    delta ~ normal(0, 2);

    // ##########
    // Non Hierarchical boundary parameters
    // ##########
    alpha_pre_acc ~ normal(0,0.2);  
    alpha_ne_pre_acc ~ normal(0, 0.1);
    

    // ##########
    // Participant-level DDM parameter priors
    // ##########
    for (p in 1:n_participants) {

        // Participant-level non-decision time
        participants_ter[p] ~ normal(ter, ter_sd) T[0, .3];

        // Participant-level boundary parameter (speed-accuracy tradeoff)
        participants_alpha[p] ~ normal(alpha, alpha_sd) T[0, 3];
                
        participants_alpha_ne[p] ~ normal(alpha_ne, alpha_ne_sd);

        //Participant-level drift rate
        participants_delta[p] ~ normal(delta, delta_sd);
                        
    }
    int grainsize = 100;     
    target += reduce_sum(partial_diffusion_lpdf, y, grainsize, participants_alpha, participants_alpha_ne, alpha_pre_acc, alpha_ne_pre_acc, participants_ter, 0.5, participants_delta, pre_ne, pre_acc, participant);         
}

generated quantities { 
   vector[N] log_lik;     
    
   // Wiener likelihood
    for (i in 1:N) {
        // Log density for DDM process
         log_lik[i] = log_likelihood_diffusion_lpdf(y[i] | participants_alpha[participant[i]], participants_alpha_ne[participant[i]], alpha_pre_acc, alpha_ne_pre_acc, participants_ter[participant[i]], 0.5, participants_delta[participant[i]], pre_ne[i], pre_acc[i]);
    }
}
