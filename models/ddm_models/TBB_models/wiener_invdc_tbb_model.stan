functions {
    /* Wiener diffusion log-PDF for a single response (adapted from brms 1.10.2)
    * Arguments:
    *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    *   invdc: invdc separation parameter > 0
    *   ndt: non-decision time parameter > 0
    *   bias: initial bias parameter in [0, 1]
    *   drift: drift rate parameter
    * Returns:
    *   a scalar to be added to the log posterior
    */
    real diffusion_lpdf(real Y, real invdc, real ndt, real bias, real drift) {
        if (abs(Y) > ndt) {
            if (Y >= 0) {
                return wiener_lpdf( abs(Y) | invdc, ndt, bias, drift*invdc );
            } else {
                return wiener_lpdf( abs(Y) | invdc, ndt, 1-bias, -drift*invdc);
            }
        } else {
            return wiener_lpdf( ndt | invdc, ndt, bias, drift*invdc );
        }
    }

    real partial_diffusion_lpdf(
        array[] real y_slice,
        int start,
        int end,
        vector invdc,
        vector invdc_cond,
        vector invdc_ne,
        real invdc_pre_acc,
        vector invdc_ne_pre_acc,
        real invdc_ne_cond,
        real invdc_pre_acc_cond,
        real invdc_ne_pre_acc_cond,
        vector ndt,
        real bias,
        vector drift,
        vector drift_cond,
        vector drift_ne,
        real drift_pre_acc,
        vector drift_ne_pre_acc,
        real drift_ne_cond,
        real drift_pre_acc_cond,
        real drift_ne_pre_acc_cond,
        vector condition,
        vector pre_ne,
        vector pre_acc,
        array[] int participant
    ) {
        int n_trials = end - start + 1;
        int global_index = start;
        vector[n_trials] partial_sum_level_likelihood;

        for (t in 1:n_trials) {
            partial_sum_level_likelihood[t] = diffusion_lpdf(y_slice[t] | invdc[participant[global_index]]  + invdc_cond[participant[global_index]]*condition[global_index] + invdc_ne[participant[global_index]]*pre_ne[global_index] +
                    invdc_pre_acc*pre_acc[global_index] + invdc_ne_pre_acc[participant[global_index]]*pre_ne[global_index]*pre_acc[global_index] + invdc_ne_cond*pre_ne[global_index]*condition[global_index] + invdc_pre_acc_cond*pre_acc[global_index]*condition[global_index] +
                    invdc_ne_pre_acc_cond*pre_ne[global_index]*pre_acc[global_index]*condition[global_index] , ndt[participant[global_index]], bias,
                    drift[participant[global_index]] + drift_cond[participant[global_index]]*condition[global_index] + drift_ne[participant[global_index]]*pre_ne[global_index] + drift_pre_acc*pre_acc[global_index] + drift_ne_pre_acc[participant[global_index]]*pre_ne[global_index]*pre_acc[global_index] +
                    drift_ne_cond*pre_ne[global_index]*condition[global_index] + drift_pre_acc_cond*pre_acc[global_index]*condition[global_index] + drift_ne_pre_acc_cond*pre_ne[global_index]*pre_acc[global_index]*condition[global_index]);
            global_index = global_index + 1;
        }
        return(sum(partial_sum_level_likelihood));
    }

    real log_likelihood_diffusion_lpdf(
        real y,
        real invdc,
        real invdc_cond,
        real invdc_ne,
        real invdc_pre_acc,
        real invdc_ne_pre_acc,
        real invdc_ne_cond,
        real invdc_pre_acc_cond,
        real invdc_ne_pre_acc_cond,
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
        return diffusion_lpdf(y | invdc  + invdc_cond*condition + invdc_ne*pre_ne + invdc_pre_acc*pre_acc + invdc_ne_pre_acc*pre_ne*pre_acc + invdc_ne_cond*pre_ne*condition + invdc_pre_acc_cond*pre_acc*condition + invdc_ne_pre_acc_cond*pre_ne*pre_acc*condition, ndt, bias, drift + drift_cond*condition + drift_ne*pre_ne + drift_ne_cond*pre_ne*condition + drift_pre_acc*pre_acc + drift_pre_acc_cond*pre_acc*condition + drift_ne_pre_acc*pre_ne*pre_acc + drift_ne_pre_acc_cond*pre_ne*pre_acc*condition);
    }

    real estimate_dc(
        real invarsigma,
        real invarsigma_cond
    ) {
        return (1/(invarsigma + invarsigma_cond*(1)) + 1/(invarsigma + invarsigma_cond*(-1)))/2;
    }

    real estimate_dc_cond(
        real invarsigma,
        real invarsigma_cond
    ) {
        return ((1/(invarsigma + invarsigma_cond*(1))) - (1/(invarsigma + invarsigma_cond*(-1))))/2;
    }

    real estimate_dc_ne(
        real invarsigma,
        real invarsigma_cond,
        real invarsigma_ne
    ) {
        return (1/(invarsigma + invarsigma_cond*(1) + invarsigma_ne*1)) - (1/(invarsigma + invarsigma_cond*(1)));
    }

    real estimate_dc_pre_acc(
        real invarsigma,
        real invarsigma_cond,
        real invarsigma_pre_acc
    ) {
        return ((1/(invarsigma + invarsigma_cond*(1) + invarsigma_pre_acc*1)) - (1/(invarsigma + invarsigma_cond*(1) + invarsigma_pre_acc*(-1))))/2;
    }

    real estimate_dc_pre_acc_cond(
        real invarsigma,
        real invarsigma_cond,
        real invarsigma_pre_acc,
        real invarsigma_pre_acc_cond

    ) {
        return 0.25 * ((1/(invarsigma + invarsigma_cond + invarsigma_pre_acc + invarsigma_pre_acc_cond)) - (1/(invarsigma + invarsigma_cond - invarsigma_pre_acc - invarsigma_pre_acc_cond)) - (1/(invarsigma - invarsigma_cond + invarsigma_pre_acc - invarsigma_pre_acc_cond)) + (1/(invarsigma - invarsigma_cond - invarsigma_pre_acc + invarsigma_pre_acc_cond)));
    }

    real estimate_dc_ne_pre_acc(
        real invarsigma,
        real invarsigma_cond,
        real invarsigma_ne,
        real invarsigma_pre_acc,
        real invarsigma_ne_pre_acc
    ) {
        return 0.5 * ((1/(invarsigma + invarsigma_cond + invarsigma_ne + invarsigma_pre_acc + invarsigma_ne_pre_acc)) - (1/(invarsigma + invarsigma_cond + invarsigma_ne - invarsigma_pre_acc - invarsigma_ne_pre_acc)) - (1/(invarsigma + invarsigma_cond + invarsigma_pre_acc)) + (1/(invarsigma + invarsigma_cond - invarsigma_pre_acc)));
    }

    real estimate_dc_ne_cond(
        real invarsigma,
        real invarsigma_cond,
        real invarsigma_ne,
        real invarsigma_ne_cond
    ){
        return 0.5 * ((1/(invarsigma + invarsigma_cond + invarsigma_ne + invarsigma_ne_cond)) - (1/(invarsigma - invarsigma_cond + invarsigma_ne - invarsigma_ne_cond)) - (1/(invarsigma + invarsigma_cond)) + (1/(invarsigma - invarsigma_cond)));
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
    vector<lower=0, upper=0.3>[n_participants] participants_ter;
    vector<lower=0, upper=3>[n_participants] participants_invarsigma;
    vector[n_participants] participants_invarsigma_cond;
    vector[n_participants] participants_invarsigma_ne;
    vector[n_participants] participants_invarsigma_ne_pre_acc;

    vector[n_participants] participants_delta;
    vector[n_participants] participants_delta_cond;
    vector[n_participants] participants_delta_ne;
    vector[n_participants] participants_delta_ne_pre_acc;

    real<lower=0> ter;
    real<lower=0, upper=3> invarsigma;
    real invarsigma_cond;
    real invarsigma_ne;
    real invarsigma_ne_pre_acc;
    real delta;
    real delta_cond;
    real delta_ne;
    real delta_ne_pre_acc;

    real<lower=0> ter_sd;
    real<lower=0> invarsigma_sd;
    real<lower=0> invarsigma_cond_sd;
    real<lower=0> invarsigma_ne_sd;
    real<lower=0> invarsigma_ne_pre_acc_sd;
    real<lower=0> delta_sd;
    real<lower=0> delta_cond_sd;
    real<lower=0> delta_ne_sd;
    real<lower=0> delta_ne_pre_acc_sd;

    // Population-level effects
    real invarsigma_pre_acc;
    real invarsigma_ne_cond;
    real invarsigma_pre_acc_cond;
    real invarsigma_ne_pre_acc_cond;

    real delta_pre_acc;
    real delta_ne_cond;
    real delta_pre_acc_cond;
    real delta_ne_pre_acc_cond;
}

model {

    // ##########
    // Between-participant variability priors
    // ##########
    ter_sd ~ gamma(.3,1);
    invarsigma_sd ~ gamma(1,1);
    invarsigma_cond_sd ~ gamma(1,1);
    invarsigma_ne_sd ~ normal(0,.5);
    invarsigma_ne_pre_acc_sd ~ normal(0,.5);
    delta_sd ~ gamma(1,1);
    delta_cond_sd ~ gamma(1,1);
    delta_ne_sd ~ normal(0,.5);
    delta_ne_pre_acc_sd ~ normal(0,.5);

    // ##########
    // Hierarchical parameters priors
    // ##########
    ter ~ normal(.1, .2) T[0, .3];
    invarsigma ~ normal(1, 1) T[0, 3];
    invarsigma_cond ~ normal(0, 1);
    invarsigma_ne ~ normal(0, 0.2);
    invarsigma_ne_pre_acc ~ normal(0, 0.2);
    delta ~ normal(0, 2);
    delta_cond ~ normal(0, 2);
    delta_ne ~ normal(0,.5);
    delta_ne_pre_acc ~ normal(0,.5);

    // ##########
    // Population-level invdc effects
    // ##########
    invarsigma_pre_acc ~ normal(0,0.2);
    invarsigma_ne_cond ~ normal(0, 0.2);
    invarsigma_pre_acc_cond ~ normal(0,0.2);
    invarsigma_ne_pre_acc_cond ~ normal(0, 0.2);

    // ##########
    // Population-level drift effects
    // ##########
    delta_pre_acc ~ normal(0, .5);
    delta_ne_cond ~ normal(0, .5);
    delta_pre_acc_cond ~ normal(0,.5);
    delta_ne_pre_acc_cond ~ normal(0, .5);


    // ##########
    // Participant-level DDM parameter priors
    // ##########
    for (p in 1:n_participants) {

        // Participant-level non-decision time
        participants_ter[p] ~ normal(ter, ter_sd) T[0, .3];

        // Participant-level invdc parameter
        participants_invarsigma[p] ~ normal(invarsigma, invarsigma_sd) T[0, 3];
        participants_invarsigma_cond[p] ~ normal(invarsigma_cond, invarsigma_cond_sd);
        participants_invarsigma_ne[p] ~ normal(invarsigma_ne, invarsigma_ne_sd);
        participants_invarsigma_ne_pre_acc[p] ~ normal(invarsigma_ne_pre_acc, invarsigma_ne_pre_acc_sd);


        //Participant-level drift rate
        participants_delta[p] ~ normal(delta, delta_sd);
        participants_delta_cond[p] ~ normal(delta_cond, delta_cond_sd);
        participants_delta_ne[p] ~ normal(delta_ne, delta_ne_sd);
        participants_delta_ne_pre_acc[p] ~ normal(delta_ne_pre_acc, delta_ne_pre_acc_sd);

     }

    int grainsize = 100;
    target += reduce_sum(partial_diffusion_lpdf, y, grainsize, participants_invarsigma, participants_invarsigma_cond, participants_invarsigma_ne, invarsigma_pre_acc, participants_invarsigma_ne_pre_acc, invarsigma_ne_cond, invarsigma_pre_acc_cond, invarsigma_ne_pre_acc_cond, participants_ter, 0.5,
        participants_delta, participants_delta_cond, participants_delta_ne, delta_pre_acc, participants_delta_ne_pre_acc, delta_ne_cond, delta_pre_acc_cond, delta_ne_pre_acc_cond, condition, pre_ne, pre_acc, participant);
}

generated quantities {
    vector[N] log_lik;

    real varsigma;
    real varsigma_cond;
    real varsigma_ne;
    real varsigma_pre_acc;
    real varsigma_ne_pre_acc;
    real varsigma_ne_cond;
    real varsigma_pre_acc_cond;
    real varsigma_ern;
    real varsigma_crn;

    vector[n_participants] participants_varsigma;
    vector[n_participants] participants_varsigma_cond;
    vector[n_participants] participants_varsigma_ne;
    vector[n_participants] participants_varsigma_ne_pre_acc;
    vector[n_participants] participants_varsigma_ern;
    vector[n_participants] participants_varsigma_crn;

    real delta_ern;
    real delta_crn;  

    vector[n_participants] participants_delta_ern;
    vector[n_participants] participants_delta_crn;

    // group-level varsigma parameters
    varsigma = estimate_dc(invarsigma, invarsigma_cond);
    varsigma_cond = estimate_dc_cond(invarsigma, invarsigma_cond);
    varsigma_ne = estimate_dc_ne(invarsigma, invarsigma_cond, invarsigma_ne);
    varsigma_pre_acc = estimate_dc_pre_acc(invarsigma, invarsigma_cond, invarsigma_pre_acc);
    varsigma_ne_pre_acc = estimate_dc_ne_pre_acc(invarsigma, invarsigma_cond, invarsigma_ne, invarsigma_pre_acc, invarsigma_ne_pre_acc);
    varsigma_ne_cond = estimate_dc_ne_cond(invarsigma, invarsigma_cond, invarsigma_ne, invarsigma_ne_cond);
    varsigma_pre_acc_cond = estimate_dc_pre_acc_cond(invarsigma, invarsigma_cond, invarsigma_pre_acc, invarsigma_pre_acc_cond);
    varsigma_ern = varsigma_ne + varsigma_ne_pre_acc*(-1);
    varsigma_crn = varsigma_ne + varsigma_ne_pre_acc*(1);

    // group-level delta parameters
    delta_ern = delta_ne + delta_ne_pre_acc*(-1);
    delta_crn = delta_ne + delta_ne_pre_acc*(1); 

    // individual-level parameters
    for (p in 1:n_participants) {
        participants_varsigma[p] = estimate_dc(participants_invarsigma[p], participants_invarsigma_cond[p]);
        participants_varsigma_cond[p] = estimate_dc_cond(participants_invarsigma[p], participants_invarsigma_cond[p]);
        participants_varsigma_ne[p] = estimate_dc_ne(participants_invarsigma[p], participants_invarsigma_cond[p], participants_invarsigma_ne[p]);
        participants_varsigma_ne_pre_acc[p] = estimate_dc_ne_pre_acc(participants_invarsigma[p], participants_invarsigma_cond[p], participants_invarsigma_ne[p], invarsigma_pre_acc, participants_invarsigma_ne_pre_acc[p]);
        participants_varsigma_ern[p] = participants_varsigma_ne[p] + participants_varsigma_ne_pre_acc[p]*(-1);
        participants_varsigma_crn[p] = participants_varsigma_ne[p] + participants_varsigma_ne_pre_acc[p]*(1);

        participants_delta_ern[p] = participants_delta_ne[p] + participants_delta_ne_pre_acc[p]*(-1);
        participants_delta_crn[p] = participants_delta_ne[p] + participants_delta_ne_pre_acc[p]*(1);
    }


   // Wiener likelihood
    for (i in 1:N) {
        // Log density for DDM process
         log_lik[i] = log_likelihood_diffusion_lpdf(y[i] | participants_invarsigma[participant[i]], participants_invarsigma_cond[participant[i]], participants_invarsigma_ne[participant[i]], invarsigma_pre_acc, participants_invarsigma_ne_pre_acc[participant[i]], invarsigma_ne_cond, invarsigma_pre_acc_cond, invarsigma_ne_pre_acc_cond, participants_ter[participant[i]], 0.5, participants_delta[participant[i]], participants_delta_cond[participant[i]], participants_delta_ne[participant[i]], delta_pre_acc, participants_delta_ne_pre_acc[participant[i]], delta_ne_cond, delta_pre_acc_cond, delta_ne_pre_acc_cond, condition[i], pre_ne[i], pre_acc[i]);
    }
}