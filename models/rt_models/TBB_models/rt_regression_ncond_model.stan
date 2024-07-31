functions{
    real partial_normal_lpdf(
        array[] real y_slice,
        int start,
        int end,
        vector mu,
        vector ne, 
        real acc,
        vector ne_acc, 
        real sigma,
        vector pre_ne,
        vector pre_acc,
        array[] int participant
    ) {
        int n_trials = end - start + 1;
        int global_index = start;
        vector[n_trials] partial_sum_level_likelihood;
        
        for (t in 1:n_trials) {
            partial_sum_level_likelihood[t] = normal_lpdf(y_slice[t] | mu[participant[global_index]] + ne[participant[global_index]]*pre_ne[global_index] + acc*pre_acc[global_index] + ne_acc[participant[global_index]]*pre_ne[global_index]*pre_acc[global_index], sigma);

            global_index = global_index + 1;
        }
        return(sum(partial_sum_level_likelihood));
    }
}

data {
    int<lower=1> N; // Number of trial-level observations
    int<lower=1> n_conditions; // Number of conditions (congruent and incongruent)
    int<lower=1> n_participants; // Number of participants

    array[N] real rt;
    vector[N] condition; // Contrast coded condition: -1 for erroneous and 1 for correct response respectively
    vector[N] pre_acc; // Contrast coded accuracy on previous trial
    vector[N] pre_ne; // centered correct/error negativity on previous trial
    array[N] int<lower=1> participant; // Participant index
}

parameters {
    vector<lower=0.1, upper=0.7>[n_participants] participants_mu;
    vector[n_participants] participants_ne;
    vector[n_participants] participants_ne_acc;  
        
    real<lower=0.1, upper=0.7> mu; // Hierarchical intercept for rt
    real ne; // Hierarchical effect of pre-trial eeg
    real ne_acc; // Hierarchical interaction effect
    
    real<lower=0> mu_sd; // Between-participants variability in intercept
    real<lower=0> ne_sd; // Between-participants variability in effect of pre-trial eeg
    real<lower=0> ne_acc_sd; // Between-participants variability in interaction effect
    
    // Non Hierarchical
    real acc;
    
    // sigma
    real<lower = 0> sigma;
}

model {

    // ##########
    // Between-participant variability priors
    // ##########
    mu_sd ~ gamma(.2,1);
    ne_sd ~ gamma(.2,1);
    ne_acc_sd ~ gamma(.2,1);

    // ##########
    // Hierarchical parameter priors
    // ##########
    mu ~ normal(.3, .5) T[.1, .7];
    ne ~ normal(0, .2); 
    ne_acc ~ normal(0, .2);

    // ##########
    // Non Hierarchical parameter priors
    // ##########
    acc ~ normal(0, .2); 
    
    // ##########
    // Sigma prior
    // ##########
    sigma ~ gamma(.2,1);


    // ##########
    // Participant-level parameter priors
    // ##########
    for (p in 1:n_participants) {

        participants_mu[p] ~ normal(mu, mu_sd) T[.1, .7];
        participants_ne[p] ~ normal(ne, ne_sd);
        participants_ne_acc[p] ~ normal(ne_acc, ne_acc_sd);
      
    }
    int grainsize = 100;     
    target += reduce_sum(partial_normal_lpdf, rt, grainsize, participants_mu, participants_ne, acc, participants_ne_acc, sigma, pre_ne, pre_acc, participant);        
}

generated quantities { 
   vector[N] log_lik; 
    real ern;
    real crn;

    vector[n_participants] participants_ern;
    vector[n_participants] participants_crn;

    // group-level parameters
    ern = ne + ne_acc*(-1);
    crn = ne + ne_acc;

    // individual-level parameters
    for (p in 1:n_participants) {
        participants_ern[p] = participants_ne[p] + participants_ne_acc[p]*(-1);
        participants_crn[p] = participants_ne[p] + participants_ne_acc[p];
    }


   // likelihood
    for (i in 1:N) {
    
         log_lik[i] = normal_lpdf(rt[i] | participants_mu[participant[i]] + participants_ne[participant[i]]*pre_ne[i] + acc*pre_acc[i] + participants_ne_acc[participant[i]]*pre_ne[i]*pre_acc[i], sigma);
    }
}
