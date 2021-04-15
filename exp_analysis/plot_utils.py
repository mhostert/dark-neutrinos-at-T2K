from functools import reduce
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages
import os

def set_plot_title(ax, selection_query, m4mz, exp_analysis_obj, smoothing_pars):
    ax.set_title(f'selection = {selection_query} @ $m_4={m4mz[0]}$ GeV, $m_{{Z^\prime}}={m4mz[1]}$ GeV\n '\
    f'{exp_analysis_obj.hierarchy} {exp_analysis_obj.D_or_M}\n '\
    f'smoothing pars = {smoothing_pars[0]} GeV, {smoothing_pars[1]} GeV')

def kde_variable_plot(var, range, bins, m4mz, exp_analysis_obj, smoothing_pars=[0.005, 0.05], selection_query='no_cuts', cumulative=False, existing_axis=None):
    assert m4mz in list(exp_analysis_obj.dfs.keys())
    
    selection_weights = exp_analysis_obj.df_base.eval(selection_query)
    kde_weights = exp_analysis_obj.kde_on_a_point(m4mz, smoothing_pars)
    total_weight = selection_weights * kde_weights

    kde_prediction, bin_edges = np.histogram(exp_analysis_obj.df_base[var],
             range=range,
             bins=bins,
             weights=total_weight,
            )
    kde_errors2 = np.histogram(exp_analysis_obj.df_base[var],
                 range=range,
                 bins=bins,
                 weights=total_weight**2,
                )[0]
    if cumulative:
        kde_prediction = np.cumsum(kde_prediction)
        kde_errors2 = np.cumsum(kde_errors2)
    kde_errors = np.sqrt(kde_errors2)

    # plotting
    if existing_axis is None:
        fsize = 11
        fig = plt.figure()
        axes_form = [0.14,0.15,0.82,0.76]
        ax = fig.add_axes(axes_form)
    else:
        ax = existing_axis
    ax.plot(bin_edges,
             np.append(kde_prediction, [0]),
             ds='steps-post',
             label=f'kde prediction: {kde_prediction.sum():.2g} '\
                f'$\pm$ {100*np.sqrt(kde_errors2.sum())/kde_prediction.sum():.2g}%')

    for edge_left, edge_right, pred, err in zip(bin_edges[:-1], bin_edges[1:], kde_prediction, kde_errors):
        ax.add_patch(
            patches.Rectangle(
            (edge_left, pred-err),
            edge_right-edge_left,
            2 * err,
            hatch="\\\\\\\\\\",
            fill=False,
            linewidth=0,
            alpha=0.4,
            )
        )

    ax.legend(frameon=False, loc='best')
    set_plot_title(ax, selection_query, m4mz, exp_analysis_obj, smoothing_pars)
    ax.set_xlabel(f'{var}')
    ax.set_ylabel(f'Number of entries')
    return ax


def kde_to_noscan_comparison(var, range, bins, m4mz, exp_analysis_obj, smoothing_pars=[0.005, 0.05], selection_query='no_cuts', cumulative=False, existing_axis=None):
    assert m4mz in list(exp_analysis_obj.dfs.keys())
    no_scan = exp_analysis_obj.dfs[m4mz]
    selection_weights = no_scan.eval(selection_query)
    actual_weights = no_scan['actual_weight']
    total_weights = selection_weights * actual_weights

    no_scan_pred, bin_edges = np.histogram(no_scan[var],
                                range=range,
                                bins=bins,
                                weights=total_weights,
                                )
    no_scan_pred_err = np.histogram(no_scan[var],
                                range=range,
                                bins=bins,
                                weights=total_weights**2,
                                )[0]
    if cumulative:
        no_scan_pred = np.cumsum(no_scan_pred)
        no_scan_pred_err = np.cumsum(no_scan_pred_err)

    if existing_axis is None:
        fsize = 11
        fig = plt.figure()
        axes_form = [0.14,0.15,0.82,0.76]
        ax = fig.add_axes(axes_form)
    else:
        ax = existing_axis

    # first KDE histogram with error bars
    kde_variable_plot(var,
                      range,
                      bins,
                      m4mz,
                      exp_analysis_obj,
                      smoothing_pars=smoothing_pars,
                      selection_query=selection_query,
                      cumulative=cumulative,
                      existing_axis=ax)
    
    # now generated prediction
    ax.errorbar((bin_edges[1:]+bin_edges[:-1])/2, no_scan_pred, 
                yerr=np.sqrt(no_scan_pred_err),
                fmt='k.',
                label=f'no scanning: {no_scan_pred.sum():.2g} '\
                f'$\pm$ {100*np.sqrt(no_scan_pred_err.sum())/no_scan_pred.sum():.2g}%')

    ax.set_ylim(bottom=0,top=ax.get_ylim()[1])
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc='best')
    set_plot_title(ax, selection_query, m4mz, exp_analysis_obj, smoothing_pars)


def comparison_plot_models(var, range, bins, m4mz, exp_analysis_objs, existing_axes=None, smoothing_pars=(0.01,0.01), selection_query='no_cuts', cumulative=False):
    fig, axes = plt.subplots(nrows=1, ncols=len(exp_analysis_objs), figsize = (len(exp_analysis_objs)*5, 4))
    for ax, exp_analysis_obj in zip(axes, exp_analysis_objs):
        kde_to_noscan_comparison(var,
                                range=range, bins=bins,
                                m4mz=m4mz,
                                exp_analysis_obj=exp_analysis_obj,
                                smoothing_pars=smoothing_pars,
                                selection_query=selection_query,
                                cumulative=cumulative,
                                existing_axis=ax,
                                )
    return fig, axes

def comparison_plot_cuts(var, range, bins, m4mz, exp_analysis_obj, selection_queries, smoothing_pars=(0.01,0.01), cumulative=False):
    fig, axes = plt.subplots(nrows=1, ncols=len(selection_queries), figsize = (len(selection_queries)*5, 4))
    for ax, selection_query in zip(axes, selection_queries):
        kde_to_noscan_comparison(var,
                                range=range, bins=bins,
                                m4mz=m4mz,
                                exp_analysis_obj=exp_analysis_obj,
                                smoothing_pars=smoothing_pars,
                                selection_query=selection_query,
                                cumulative=cumulative,
                                existing_axis=ax,
                                )
    return fig, axes

# def kde_to_noscan_comparison_batch(vars_ranges_binss, m4mzs, exp_analysis_objs, selection_queries, smoothing_parss=(0.01,0.01), cumulatives=False):
#     features = [vars_ranges_binss, m4mzs, exp_analysis_objs, selection_queries, smoothing_parss, cumulatives]
#     features = [feature if type(feature) is list else [feature] for feature in features]
#     n_plots = reduce((lambda x, y: len(x)*len(y)), features)
    
#     fig, axes = plt.subplots(nrows=1, ncols=len(selection_queries), figsize = (len(selection_queries)*5, 4))

#     for var_range_bins, range, bins, m4mz, exp_analysis_obj, selection_query, smoothing_pars, cumulative in product(features):
#         kde_to_noscan_comparison(var=var_range_bins[0],
#                                 range=var_range_bins[1],
#                                 bins=var_range_bins[2],
#                                 m4mz=m4mz,
#                                 exp_analysis_obj=exp_analysis_obj,
#                                 smoothing_pars=smoothing_pars,
#                                 selection_query=selection_query,
#                                 cumulative=cumulative,
#                                 existing_axis=ax,
#                                 )
#     return fig, axes

def weighted_efficiency(num_weights, anti_num_weights):
    den_weights = num_weights + anti_num_weights

    den_sum = den_weights.sum()

    num_sum = num_weights.sum()
    num2_sum = (num_weights**2).sum()

    anti_num_sum = anti_num_weights.sum()
    anti_num2_sum = (anti_num_weights**2).sum()

    eff = num_sum/den_sum
    eff_err = np.sqrt(num_sum**2 * anti_num2_sum + anti_num_sum**2 * num2_sum)/den_sum**2
    return eff, eff_err
    
def kde_efficiency(num_selection_query, den_selection_query, m4mz, exp_analysis_obj, smoothing_pars=[0.005, 0.05]):
    num_selection_weights = exp_analysis_obj.df_base.eval(" & ".join([num_selection_query, den_selection_query]))
    anti_num_selection_weights = exp_analysis_obj.df_base.eval(" & ".join(["~"+num_selection_query, den_selection_query]))
    kde_weights = exp_analysis_obj.kde_on_a_point(m4mz, smoothing_pars)
    num_weights = num_selection_weights * kde_weights
    anti_num_weights = anti_num_selection_weights * kde_weights

    return weighted_efficiency(num_weights, anti_num_weights)

def no_scan_efficiency(num_selection_query, den_selection_query, m4mz, exp_analysis_obj):
    assert m4mz in list(exp_analysis_obj.dfs.keys())
    no_scan = exp_analysis_obj.dfs[m4mz]
    num_selection_weights = no_scan.eval(" & ".join([num_selection_query, den_selection_query]))
    anti_num_selection_weights = no_scan.eval(" & ".join(["~"+num_selection_query, den_selection_query]))
    actual_weights = no_scan['actual_weight']
    num_weights = num_selection_weights * actual_weights
    anti_num_weights = anti_num_selection_weights * actual_weights

    return weighted_efficiency(num_weights, anti_num_weights)

def kde_no_scan_efficiency_cut_list(num_selection_queries, den_selection_queries, exp_analysis_obj, m4mz,smoothing_pars=[0.005, 0.05]):
    assert type(den_selection_queries) is str or len(den_selection_queries) == len(num_selection_queries)
    if type(den_selection_queries) is str:
        den_selection_queries = [den_selection_queries] * len(num_selection_queries)
    kde_eff = []
    kde_eff_err = []
    no_scan_eff = []
    for num_selection_query, den_selection_query in zip(num_selection_queries, den_selection_queries):
        kde_aux = kde_efficiency(num_selection_query,
                den_selection_query,
                m4mz=m4mz,
                exp_analysis_obj=exp_analysis_obj,
                smoothing_pars=smoothing_pars
                )
        kde_eff.append(kde_aux[0])
        kde_eff_err.append(kde_aux[1])
        no_scan_aux = no_scan_efficiency(num_selection_query,
                den_selection_query,
                m4mz=m4mz,
                exp_analysis_obj=exp_analysis_obj
                )
        no_scan_eff.append(no_scan_aux[0])

    kde_eff = np.array(kde_eff)
    kde_eff_err = np.array(kde_eff_err)
    no_scan_eff = np.array(no_scan_eff)

    plt.plot(kde_eff, label='KDE')
    plt.fill_between(range(len(num_selection_queries)),
                     kde_eff-kde_eff_err, kde_eff+kde_eff_err,
                     alpha=0.4,
                     color=plt.gca().lines[-1].get_color(), 
                     interpolate=True)
    plt.plot(no_scan_eff, '.', label='no scan')
    plt.legend(frameon=False)
    plt.xlabel('selection cut')

    plt.xticks(ticks=range(len(num_selection_queries)),
               labels=num_selection_queries,
               rotation=30)

    plt.ylabel('efficiency')

    set_plot_title(plt.gca(), den_selection_queries[0], m4mz, exp_analysis_obj, smoothing_pars)

    return kde_eff/no_scan_eff, kde_eff_err/no_scan_eff

def kde_no_scan_efficiency_plot_grid(num_selection_query, den_selection_query, exp_analysis_obj, smoothing_pars=[0.005, 0.05]):

    no_scan_eff, no_scan_eff_err = no_scan_efficiency(num_selection_query, den_selection_query, m4mz, exp_analysis_obj)
    kde_eff, kde_eff_err = kde_efficiency(num_selection_query, den_selection_query, m4mz, exp_analysis_obj, smoothing_pars)

    return kde_eff/no_scan_eff, kde_eff_err/no_scan_eff





# def batch_comparison_cutlevels(pdffilepath, exp_analyses, m4mz, smooth=(0.01,0.01), sel_criterion=False, selection=True, variables=False, var_range=False, bins=False):
#     '''streamlining the 4 panel plots -- currently only one hierarchy at a time'''

#     if not os.path.isdir(os.path.basename(pdffilepath)):
#         os.makedirs(os.path.basename(pdffilepath))

#     # create pdf page...
#     pdf = PdfPages(pdffilepath)

#     if not variables:
#         variables = [ ('ee_energy', ''),
#                     ('ee_theta', ''),
#                     ('ee_mass', ''),
#                     ('ee_energy_asymetry', ''),
#                     ('em_beam_theta', ''),
#                     ('ep_beam_theta', ''),
#                     ('experimental_t', '')]

#     if not var_range:
#         var_range = [(0,1.0),
#                     (0,np.pi/2),
#                     (0,m4mz[0]),
#                     (-1,1.0),
#                     (0,np.pi/2),
#                     (0,np.pi/2),
#                     (0,0.06)]
#     if not bins:                
#         bins = [10,
#                 10,
#                 10,
#                 10,
#                 10,
#                 10,
#                 10]
#     elif np.size(bins)==1:
#         bins = np.ones(np.size(variables))*bins
    
#     if not sel_criterion:                
#         sel_criterion = np.full(4,'no_cuts')
#     elif np.size(sel_criterion)==1:
#         sel_criterion = np.full(4,sel_criterion)

#     ###############
#     # all variables
#     for i in range(np.shape(variables)[0]):
#         ################
#         # all four panels
#         fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))
#         for k in range(4):
#             kde_to_noscan_comparison(var1=variables[i][0], var2=variables[i][1], 
#                                     range=var_range[i], bins=bins[i], 
#                                     m4mz=m4mz, sel_criterion=sel_criterion[k],
#                                     exp_analysis_obj=exp_analyses[k],
#                                     smoothing_pars=smooth, axis=axes[k], selection=selection)
#         plt.tight_layout(); pdf.savefig(fig)
#     plt.tight_layout()
#     pdf.close()

#######################################################
# # dirty function to plot everything...
# def batch_comparison_light_heavy(pdffilepath, exp_analyses, m4mzheavy, m4mzlight, smooth=(0.01,0.01), selection=True):
    
#     if not os.path.isdir(os.path.basename(pdffilepath)):
#         os.makedirs(os.path.basename(pdffilepath))

#     # create pdf page...
#     pdf = PdfPages(pdffilepath)

#     ######################
#     bins = 10
#     var1='ee_energy'
#     var2=''
#     varmin=0; varmax=1.0
#     fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

#     # Heavy
#     exp_analyses_h=exp_analyses[:2]
#     batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     # light
#     exp_analyses_l=exp_analyses[2:]
#     batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     plt.tight_layout(); pdf.savefig(fig)

#     ######################
#     bins = 10
#     var1='ee_theta'
#     var2=''
#     varmin=0; varmax=np.pi/2
#     fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

#     # Heavy
#     exp_analyses_h=exp_analyses[:2]
#     batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     # light
#     exp_analyses_l=exp_analyses[2:]
#     batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     plt.tight_layout(); pdf.savefig(fig)

#     ######################
#     bins = 5
#     var1='ee_mass'
#     var2=''
#     fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

#     # Heavy
#     varmin=0; varmax=m4mzheavy[0]
#     exp_analyses_h=exp_analyses[:2]
#     batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     # light
#     varmin=0; varmax=m4mzlight[0]
#     exp_analyses_l=exp_analyses[2:]
#     batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     plt.tight_layout(); pdf.savefig(fig)

#     ######################
#     bins = 20
#     var1='ee_energy_asymetry'
#     var2=''
#     varmin=-1; varmax=1
#     fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

#     # Heavy
#     exp_analyses_h=exp_analyses[:2]
#     batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     # light
#     exp_analyses_l=exp_analyses[2:]
#     batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     plt.tight_layout(); pdf.savefig(fig)

#     ######################
#     bins = 20
#     var1='em_beam_theta'
#     var2=''
#     varmin=0; varmax=np.pi
#     fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

#     # Heavy
#     exp_analyses_h=exp_analyses[:2]
#     batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     # light
#     exp_analyses_l=exp_analyses[2:]
#     batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     plt.tight_layout(); pdf.savefig(fig)


#     ######################
#     var1='ep_beam_theta'
#     fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

#     # Heavy
#     exp_analyses_h=exp_analyses[:2]
#     batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     # light
#     exp_analyses_l=exp_analyses[2:]
#     batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     plt.tight_layout(); pdf.savefig(fig)

#     ######################
#     bins = 20
#     var1='experimental_t'
#     var2=''
#     varmin=0; varmax=0.06
#     fig,  axes = plt.subplots(nrows=1, ncols=4,figsize = (18,4))

#     # Heavy
#     exp_analyses_h=exp_analyses[:2]
#     batch_comparison_plot(axes[:2],exp_analyses_h, m4mzheavy,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     # light
#     exp_analyses_l=exp_analyses[2:]
#     batch_comparison_plot(axes[2:],exp_analyses_l, m4mzlight,var1,var2,smooth=smooth,var_range=(varmin,varmax), bins=bins)

#     plt.tight_layout(); pdf.savefig(fig)

#     pdf.close()

