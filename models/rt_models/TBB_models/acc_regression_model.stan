functions{
    real partial_bernoulli_logit_lpmf(
        array[] int y_slice,
        int start,
        int end,
        vector mu,
        vector cond,
        vector ne, 
        real ne_cond,
        real acc,
        real acc_cond,
        vector ne_acc, 
        real ne_acc_cond,        
        vector condition, 
        vector pre_ne,
        vector pre_acc,
        array[] int participant
    ) {
        int n_trials = end - start + 1;
        int global_index = start;
        vector[n_trials] partial_sum_level_likelihood;
        
        for (t in 1:n_trials) {
            partial_sum_level_likelihood[t] = bernoulli_logit_lpmf(y_slice[t] | mu[participant[global_index]] + cond[participant[global_index]]*condition[global_index] + ne[participant[global_index]]*pre_ne[global_index] + ne_cond*pre_ne[global_index]*condition[global_index] + acc*pre_acc[global_index] + acc_cond*pre_acc[global_index]*condition[global_index] + ne_acc[participant[global_index]]*pre_ne[global_index]*pre_acc[global_index] + ne_acc_cond*pre_ne[global_index]*pre_acc[global_index]*condition[global_index]);

            global_index = global_index + 1;
        }
        return(sum(partial_sum_level_likelihood));
    }
}

data {
    int<lower=1> N; // Number of trial-level observations
    int<lower=1> n_conditions; // Number of conditions (congruent and incongruent)
    int<lower=1> n_participants; // Number of participants

    array[N] int y_acc;
    vector[N] condition; // Contrast coded condition: -1 for erroneous and 1 for correct response respectively
    vector[N] pre_acc; // Contrast coded accuracy on previous trial
    vector[N] pre_ne; // centered correct/error negativity on previous trial
    array[N] int<lower=1> participant; // Participant index
}

parameters {
    vector[n_participants] participants_mu;
    vector[n_participants] participants_cond;
    vector[n_participants] participants_ne;
    vector[n_participants] participants_acc; 
    vector[n_participants] participants_ne_acc;  
        
    real mu; // Hierarchical intercept for rt
    real cond; // Hierarchical effect of condition
    real ne; // Hierarchical effect of pre-trial eeg
    real acc; // Hierarchical effect of pre-trial accuracy
    real ne_acc; // Hierarchical interaction effect
    
    real<lower=0> mu_sd; // Between-participants variability in intercept
    real<lower=0> cond_sd; // Between-participants variability in effect of condition
    real<lower=0> ne_sd; // Between-participants variability in effect of pre-trial eeg
    real<lower=0> acc_sd; // Between-participants variability in effect of pre-trial accuracy
    real<lower=0> ne_acc_sd; // Between-participants variability in interaction effect
    
    // Non Hierarchical
    real ne_cond; 
    real acc_cond;
    real ne_acc_cond; 
}

model {

    // ##########
    // Between-participant variability priors
    // ##########
    mu_sd ~ gamma(1,1);
    cond_sd ~ gamma(1,1);
    ne_sd ~ gamma(1,1);
    ne_acc_sd ~ gamma(1,1);

    // ##########
    // Hierarchical parameter priors
    // ##########
    mu ~ normal(0, 1.5);
    cond ~ normal(0, .2);
    ne ~ normal(0, .2); 
    ne_acc ~ normal(0, .2);

    // ##########
    // Non Hierarchical parameter priors
    // ##########
    acc ~ normal(0, .2); 
    ne_cond ~ normal(0,0.2);  
    acc_cond ~ normal(0, 0.2);
    ne_acc_cond ~ normal(0, 0.2);
    

    // ##########
    // Participant-level parameter priors
    // ##########
    for (p in 1:n_participants) {

        participants_mu[p] ~ normal(mu, mu_sd);
        participants_cond[p] ~ normal(cond, cond_sd);
        participants_ne[p] ~ normal(ne, ne_sd);
        participants_ne_acc[p] ~ normal(ne_acc, ne_acc_sd);
      
    }
    int grainsize = 100;     
    target += reduce_sum(partial_bernoulli_logit_lpmf, y_acc, grainsize, participants_mu, participants_cond, participants_ne, ne_cond, acc, acc_cond, participants_ne_acc, ne_acc_cond, condition, pre_ne, pre_acc, participant);        
}

generated quantities { 
    // vector[N] log_lik; 
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

    // // results in a proportion space
    // real mu_prop = inv_logit(mu);
    // real cond_prop = inv_logit(mu) - inv_logit(mu - cond);
    // real ne_prop = inv_logit(mu) - inv_logit(mu - ne);
    // real pre_acc_prop = inv_logit(mu) - inv_logit(mu - acc);
    // real ne_pre_acc_prop = inv_logit(mu) - inv_logit(mu - ne_acc);
    // real ne_cond_prop = inv_logit(mu) - inv_logit(mu - ne_cond);
    // real acc_cond_prop = inv_logit(mu) - inv_logit(mu - acc_cond);
    // real ne_acc_cond_prop = inv_logit(mu) - inv_logit(mu - ne_acc_cond);
    // real ern_prop = inv_logit(mu) - inv_logit(mu - ern);
    // real crn_prop = inv_logit(mu) - inv_logit(mu - crn);


    // vector[n_participants] participants_mu_prop = inv_logit(participants_mu);
    // vector[n_participants] participants_cond_prop = inv_logit(participants_mu) - inv_logit(participants_mu - participants_cond);
    // vector[n_participants] participants_ne_prop = inv_logit(participants_mu) - inv_logit(participants_mu - participants_ne);
    // vector[n_participants] participants_ne_acc_prop = inv_logit(participants_mu) - inv_logit(participants_mu - participants_ne_acc) ;
    // vector[n_participants] participants_ern_prop = inv_logit(participants_mu) - inv_logit(participants_mu - participants_ern);
    // vector[n_participants] participants_crn_prop = inv_logit(participants_mu) - inv_logit(participants_mu - participants_crn); 


    // // likelihood
    // for (i in 1:N) {

    //      log_lik[i] = bernoulli_logit_lpmf(y_acc[i] | participants_mu[participant[i]] + participants_cond[participant[i]]*condition[i] + participants_ne[participant[i]]*pre_ne[i] + ne_cond*pre_ne[i]*condition[i] + acc*pre_acc[i] + acc_cond*pre_acc[i]*condition[i] + participants_ne_acc[participant[i]]*pre_ne[i]*pre_acc[i] + ne_acc_cond*pre_ne[i]*pre_acc[i]*condition[i]);
    // }
}
