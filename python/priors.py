from scipy import stats

sonata_cond_rt_regression_priors = {
    'participants_cond':[{
        'mean': {'loc':0, "scale":.2},
        'sd': ('gamma', {'shape':1, "scale":1}),
    }, True],
    'participants_ne':[{
        'mean': {'loc':0, "scale":.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_ne_acc':[{
        'mean': {'loc':0, "scale":.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_ern':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'participants_crn':[[{
        'mean': {'loc':0, "scale":0.2},
         'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
         'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'cond': [stats.norm(loc=0, scale=.2), False],
    'ne': [stats.norm(loc=0, scale=0.2), False],
    'ne_acc': [stats.norm(loc=0, scale=0.2), False],
    'acc': [stats.norm(loc=0, scale=0.2), False],
    'ne_cond': [stats.norm(loc=0, scale=0.2), False],
    'acc_cond': [stats.norm(loc=0, scale=0.2), False],
    'ne_acc_cond': [stats.norm(loc=0, scale=0.2), False],
    'ern': [[stats.norm(loc=0, scale=0.2), stats.norm(loc=0, scale=0.2)], False],
    'crn': [[stats.norm(loc=0, scale=0.2), stats.norm(loc=0, scale=0.2)], False],   
}

sonata_cond_acc_regression_priors = {
    'participants_cond':[{
        'mean': {'loc':0, "scale":.2},
        'sd': ('gamma', {'shape':1, "scale":1}),
    }, True],
    'participants_ne':[{
        'mean': {'loc':0, "scale":.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_ne_acc':[{
        'mean': {'loc':0, "scale":.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_ern':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'participants_crn':[[{
        'mean': {'loc':0, "scale":0.2},
         'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
         'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'cond': [stats.norm(loc=0, scale=.2), False],
    'ne': [stats.norm(loc=0, scale=0.2), False],
    'ne_acc': [stats.norm(loc=0, scale=0.2), False],
    'acc': [stats.norm(loc=0, scale=0.2), False],
    'ne_cond': [stats.norm(loc=0, scale=0.2), False],
    'acc_cond': [stats.norm(loc=0, scale=0.2), False],
    'ne_acc_cond': [stats.norm(loc=0, scale=0.2), False],
    'ern': [[stats.norm(loc=0, scale=0.2), stats.norm(loc=0, scale=0.2)], False],
    'crn': [[stats.norm(loc=0, scale=0.2), stats.norm(loc=0, scale=0.2)], False],   
}

drift_boundary_ncond_prior = {
    'participants_alpha_ne':[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_alpha_ne_pre_acc':[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_alpha_ern':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'participants_alpha_crn':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'participants_delta_ne': [{
        'mean': {'loc':0, "scale":.5},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_delta_ne_pre_acc':[{
        'mean': {'loc':0, "scale":0.5},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_delta_ern':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'participants_delta_crn':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'alpha_ne': [stats.norm(loc=0, scale=0.2), False],
    'alpha_ern': [[stats.norm(loc=0, scale=0.2), stats.norm(loc=0, scale=0.2)], False],
    'alpha_crn': [[stats.norm(loc=0, scale=0.2), stats.norm(loc=0, scale=0.2)], False],
    'delta_ne': [stats.norm(loc=0, scale=.5), False],
    'delta_ern': [[stats.norm(loc=0, scale=0.5), stats.norm(loc=0, scale=0.5)], False],
    'delta_crn': [[stats.norm(loc=0, scale=0.5), stats.norm(loc=0, scale=0.5)], False],
    
    'alpha_pre_acc': [stats.norm(loc=0, scale=0.2), False],
    'alpha_ne_pre_acc': [stats.norm(loc=0, scale=0.2), False],

   
    'delta_pre_acc': [stats.norm(loc=0, scale=.5), False],
    'delta_ne_pre_acc': [stats.norm(loc=0, scale=.5), False],
}

drift_boundary_cond_prior = {
    'participants_alpha_cond':[{
        'mean': {'loc':0, "scale":1},
        'sd': ('gamma', {'shape':1, "scale":1}),
    }, True],
    'participants_alpha_ne':[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_alpha_ne_pre_acc':[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_alpha_ern':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'participants_alpha_crn':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'participants_delta_cond': [{
        'mean': {'loc':0, "scale":2},
        'sd': ('gamma', {'shape':1, "scale":1}),
    }, True],
    'participants_delta_ne': [{
        'mean': {'loc':0, "scale":.5},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_delta_ne_pre_acc':[{
        'mean': {'loc':0, "scale":0.5},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, True],
    'participants_delta_ern':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'participants_delta_crn':[[{
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }, {
        'mean': {'loc':0, "scale":0.2},
        'sd': ('normal', {'shape':0, "scale":.5}),
    }], True],
    'alpha_cond': [stats.norm(loc=0, scale=1), False],
    'alpha_ne': [stats.norm(loc=0, scale=0.2), False],
    'alpha_ern': [[stats.norm(loc=0, scale=0.2), stats.norm(loc=0, scale=0.2)], False],
    'alpha_crn': [[stats.norm(loc=0, scale=0.2), stats.norm(loc=0, scale=0.2)], False],
    'delta_cond': [stats.norm(loc=0, scale=2), False],
    'delta_ne': [stats.norm(loc=0, scale=.5), False],
    'delta_ern': [[stats.norm(loc=0, scale=0.5), stats.norm(loc=0, scale=0.5)], False],
    'delta_crn': [[stats.norm(loc=0, scale=0.5), stats.norm(loc=0, scale=0.5)], False],
    
    'alpha_pre_acc': [stats.norm(loc=0, scale=0.2), False],
    'alpha_ne_pre_acc': [stats.norm(loc=0, scale=0.2), False],
    'alpha_ne_cond': [stats.norm(loc=0, scale=0.2), False],
    'alpha_pre_acc_cond':[stats.norm(loc=0, scale=0.2), False],
    'alpha_ne_pre_acc_cond':[stats.norm(loc=0, scale=0.2), False],
   
    'delta_pre_acc': [stats.norm(loc=0, scale=.5), False],
    'delta_ne_pre_acc': [stats.norm(loc=0, scale=.5), False],
    'delta_ne_cond': [stats.norm(loc=0, scale=.5), False],
    'delta_pre_acc_cond':[stats.norm(loc=0, scale=.5), False],
    'delta_ne_pre_acc_cond':[stats.norm(loc=0, scale=.5), False],
}