import numpy as np
import pandas as pd
import cmdstanpy
from scipy import stats
import arviz.labels as azl
import arviz as az
import math
import matplotlib.pyplot as plt
import seaborn as sns
import re

def plot_traces(
    inference_data, 
    variables, 
    save=False, 
    mapper=None, 
    path=None, 
    model=None
):
    az.style.use("arviz-doc")
    
    cm = 1/2.54
    dpi = 300
    
    az.rcParams["plot.max_subplots"] = 500
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.edgecolor"] = ".15"
    plt.rcParams["axes.linewidth"]  = 0.5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['lines.linewidth'] = .5             
    plt.rcParams['axes.titleweight'] = 'normal'

    labeller = azl.MapLabeller(var_name_map=mapper)
    
    fig = plt.figure()

    axes = az.plot_trace(
        inference_data, 
        var_names=variables, 
        divergences=None,
        figsize=(15*cm, 3.5*len(variables)*cm),
        labeller=labeller,
    )
    
    fig.tight_layout()
                         
    if save:
        plt.savefig(f'{path}/{model}/results/trace_plots/{model}_traceplots_{save}', bbox_inches='tight')

    plt.show()

def plot_posteriors(
    inference_data, 
    variables, 
    ref_val=None, 
    save=False, 
    mapper=None, 
    dpi=300, 
    path=None, 
    model=None,
    **kwargs
):
    cm = 1/2.54
    
    az.rcParams["plot.max_subplots"] = 500
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['ytick.labelsize'] = 5
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['axes.labelsize'] = 5
    plt.rcParams['axes.titlesize'] = 6
    plt.rcParams["font.size"] = 5
    plt.rcParams["axes.edgecolor"] = ".15"
    plt.rcParams["axes.linewidth"]  = 0.5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['lines.linewidth'] = .7             
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['axes.titlepad'] = .1 
    palette = sns.color_palette("colorblind")

    n_cols = 6 if len(variables) > 6 else len(variables)
    n_rows = 1 if len(variables) < 6 else math.ceil(len(variables)/6)

    labeller = azl.MapLabeller(var_name_map=mapper)
    blue = palette[0]  
    green = palette[2] 
    red = palette[3]
    
    fig = plt.figure(dpi=dpi)

    axes = az.plot_posterior(
        inference_data,
        var_names=variables,
        grid = (n_rows,n_cols),
        hdi_prob = 0.95,
        figsize=(2.5*n_cols*cm, 3*cm*n_rows),
        point_estimate='mean',
        kind='hist',
        bins=20,
        ref_val = ref_val,
        backend_kwargs = {'gridspec_kw': {'wspace':0.45, 'hspace': 0.35}, 'subplot_kw': {'box_aspect':1.1}},
        labeller=labeller,
        round_to=2,
        **kwargs
    )
    if isinstance(axes, (list, np.ndarray)):
        fig = axes.flatten()[0].get_figure()
    else:
        fig = axes.get_figure()
    fig.tight_layout()
    if save:
        plt.savefig(f'{path}/{model}/results/posterior_plots/{model}_posteriors_{save}', bbox_inches='tight')
    plt.show()

def plot_individual_bayes_factors(df, save=False, path=None, model=None):

    sns.set_palette("colorblind")
    
    g = sns.FacetGrid(
        summary_copy_random_effects, 
        col='parameter_name', 
        col_wrap=3, 
        sharey=False,
        sharex=False,
        aspect=1,
        height=6,
    )
    
    g.map(
        sns.pointplot, 
        'Bayes_factor', 
        'index', 
    )
    
    g.set_yticklabels([])
    g.set_axis_labels("Bayes factor", "ID")
    g.set_titles(col_template="{col_name}")
    
    # Add vertical lines at x-values 0.1 and 10
    def add_vertical_lines(x, color, linestyle):
        plt.axvline(x=x, color=color, linestyle=linestyle)
        
    g.map(add_vertical_lines, color='g', linestyle='--', x=0.5)
    g.map(add_vertical_lines, color='black', linestyle='--', x=1)
    g.map(add_vertical_lines, color='r', linestyle='--', x=10)
    g.map(add_vertical_lines, color='r', linestyle='--', x=5)
    
    
    g.fig.tight_layout()
    
    fig = plt.gcf()
    if save:
        plt.savefig(f'{path}/{model}/results/{model}_random_effects_bfs.png', bbox_inches='tight')
    plt.show()


def plot_random_effects_distributions(
    inference_data, 
    summary_df, 
    df, 
    parameter_name_order=None, 
    save=None,
    path=None, 
    model=None, 
    effects_titles_mapper=None, 
    n_cols=2, 
    aspect=1.9, 
    sharey=False, 
    dpi=300,
    y_lim=(-1, 1)
):
    # Set global plotting parameters
    def set_plotting_params(dpi):
        plt.rcParams.update({
            'figure.dpi': dpi,
            'ytick.labelsize': 20,
            'xtick.labelsize': 20,
            'axes.labelsize': 25,
            'axes.titlesize': 20,
            'font.size': 20,
            'axes.edgecolor': '.15',
            'axes.linewidth': 2,
            'ytick.major.size': 5,
            'ytick.major.width': 1,
        })
        sns.set_style("ticks")
    
    set_plotting_params(dpi=dpi)
    palette = sns.color_palette("colorblind")
    cm = 1 / 2.54
    
    # Add vertical lines helper
    def add_vertical_lines(x, color, linestyle):
        line = plt.axvline(x=x, color=color, linestyle=linestyle)
        plt.setp(line, zorder=1000)

    # Add horizontal lines helper
    def add_horizontal_lines(y, color, linestyle, linewidth=2):
        line = plt.axhline(y=y, color=color, linestyle=linestyle, linewidth=linewidth)
        plt.setp(line, zorder=1000)
    
    # Define colors from the palette
    blue = palette[0]
    green = palette[2]
    red = palette[3]
    dark_gray = '#36454F'
    
    # Determine parameter_name_order if not provided
    if parameter_name_order is None:
        parameter_name_order = sorted(df['parameter_name'].unique())
    
    dataset_size = next((value for key, value in inference_data.posterior.sizes.items() if 'participants' in key), None)
    width = dataset_size / 6
    height = 3.5 * (len(parameter_name_order) / n_cols)
    
    fig = plt.figure(figsize=(width * cm, height * cm), dpi=dpi)

    # Custom plotting function for coloring error bars
    def plot_effect_sizes(x, y, **kwargs):
        ax = plt.gca()
        data = kwargs.pop('data')
        errorbar = kwargs.pop('errorbar', None)
        
        sns.pointplot(
            x=x, y=y, data=data, errorbar=errorbar, color='k', alpha=1, 
            err_kws={'linewidth': 4}, markersize=4,
        )
        
        subj_idx = 0
        step = inference_data.posterior['chain'].shape[0] * inference_data.posterior['draw'].shape[0]
        for line in ax.get_lines():
            x_data, y_data = line.get_data()
            
            if len(x_data) == 2:
                parameter = data['parameter'].iloc[subj_idx * step]
                bayes_factor = data.loc[data['parameter'] == parameter, 'bayes_factor'].values[0]
                
                if ((bayes_factor < 0.33) and not np.isinf(bayes_factor)):
                    line.set_color(red)
                elif (0.33 <= bayes_factor < 3 and not np.isinf(bayes_factor)):
                    line.set_color('gray')
                elif np.isinf(bayes_factor):
                    line.set_color(green)
                else:
                    line.set_color(green)
                line.set_alpha(0.75)
                plt.setp(line, zorder=10)
                subj_idx += 1
            else:
                plt.setp(line, zorder=10000)
        
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_visible(True)
        
        pattern = re.compile(r'^participants_(.*)')
        parameter_name = pattern.match(data['parameter_name'].iloc[0]).group(1)
        population_effect = summary_df.loc[parameter_name]['Mean']
        add_horizontal_lines(y=population_effect, color=dark_gray, linestyle='--', linewidth=3)
    
    # Create FacetGrid
    g = sns.FacetGrid(
        df, 
        col='parameter_name', col_wrap=n_cols, 
        sharey=sharey, sharex=False, aspect=aspect, height=3, 
        col_order=parameter_name_order
    )
    
    g.map_dataframe(plot_effect_sizes, 'parameter', 'value', errorbar=("pi", 95))
    
    # Apply y-limits: either a single range for all or specific ranges for each facet
    if isinstance(y_lim, tuple):
        y_min, y_max = y_lim
        for ax in g.axes.flatten():
            ax.set_ylim(y_min, y_max)
    elif isinstance(y_lim, list) and len(y_lim) == len(parameter_name_order):
        for ax, y_range in zip(g.axes.flatten(), y_lim):
            y_min, y_max = y_range
            ax.set_ylim(y_min, y_max)
    else:
        raise ValueError("y_lim must be either a tuple or an array with the same length as parameter_name_order")
    
    # Set y-axis titles using the effects_titles_mapper if provided
    if effects_titles_mapper is not None:
        for ax, parameter in zip(g.axes.flat, g.col_names):
            ax.set_ylabel(effects_titles_mapper.get(parameter, parameter))
    
    # Remove facet titles
    g.set_titles(col_template="")
    
    # Add horizontal reference lines at y=0
    g.map(add_horizontal_lines, color='r', linestyle='-', linewidth=2, y=0)
    
    # Adjust layout and save the figure
    g.fig.tight_layout()
    fig = plt.gcf()
    if save is not None:
        plt.savefig(
            f'{path}/{model}/results/{model}_{save}.png', 
            bbox_inches='tight'
        )
    plt.show()