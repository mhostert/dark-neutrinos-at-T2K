from functools import reduce
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages
import os

def set_plot_title(ax=None, selection_query=None, m4mz=None, exp_analysis_obj=None, kernel=None, smoothing_pars=None, suptitle=False, same_line=False):
    if ax is not None:
        plt.sca(ax)
    title_string = ''
    if selection_query is not None:
        title_string += f'selection = {selection_query}\n'
    if m4mz is not None:
        title_string += f'@ $m_4={m4mz[0]}$ GeV, $m_{{Z^\prime}}={m4mz[1]}$ GeV\n'
    if exp_analysis_obj is not None:
        title_string += f'{exp_analysis_obj.hierarchy} {exp_analysis_obj.D_or_M}\n'
    if kernel is not None:
        title_string += f'kernel = {kernel}\n'
    if smoothing_pars is not None:
        title_string += f'smoothing pars = {smoothing_pars[0]} GeV, {smoothing_pars[1]} GeV\n'
    title_string = title_string.strip()
    if same_line:
        title_string = title_string.replace('\n', ', ')
    if not suptitle:
        plt.title(title_string)
    else:
        plt.suptitle(title_string)

def annotated_2d_plot(data, xcenters, ycenters, xlabel=None, ylabel=None, errors_to_annotate=None, colornorm='div', main_to_annotate=None, err_to_annotate=None, in_log=False, **kwargs):
    if main_to_annotate is not None:
        assert len(main_to_annotate) == len(err_to_annotate)
    if colornorm == 'div':        
        aux_norm = colors.TwoSlopeNorm(**kwargs)
        plt.pcolormesh(data.T, cmap='BrBG', norm=aux_norm)
    elif colornorm == 'normal':
        plt.pcolormesh(data.T)
    if in_log:
        xcenters = [f'{xcenter:.2g}' for xcenter in xcenters]
        ycenters = [f'{ycenter:.2g}' for ycenter in ycenters]
    plt.xticks(ticks=np.arange(0.5, len(xcenters)),
               labels=xcenters)
    plt.yticks(ticks=np.arange(0.5, len(ycenters)),
               labels=ycenters)
    for i in range(len(xcenters)):
        for j in range(len(ycenters)):
            this_value = data[i,j]
            if np.isnan(this_value):
                continue
            if main_to_annotate is None:
                text = f"{data[i,j]:.3g}"
                if errors_to_annotate is not None:
                    text += f'\n$\pm${errors_to_annotate[i,j]:.2g}'
            else:
                text = ''
                for main, err in zip(main_to_annotate, err_to_annotate):
                    text += f'{main[i,j]:.3g}\n$\pm${err[i,j]:.2g}\n'
                text = text.strip()
            plt.text(i + 0.5, j + 0.5, text, ha="center", va="center", color="k")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def kde_variable_plot(var, range, bins, m4mz, exp_analysis_obj, smoothing_pars=[0.005, 0.05], selection_query='no_cuts', cumulative=False, existing_axis=None):
    assert m4mz in list(exp_analysis_obj.dfs.keys())

    selected_df = exp_analysis_obj.df_base.query(selection_query)
    kde_weights = exp_analysis_obj.kde_on_a_point(selected_df, m4mz, smoothing_pars)

    kde_prediction, bin_edges = np.histogram(exp_analysis_obj.df_base[var],
             range=range,
             bins=bins,
             weights=kde_weights,
            )
    kde_errors2 = np.histogram(exp_analysis_obj.df_base[var],
                 range=range,
                 bins=bins,
                 weights=kde_weights**2,
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
    selected_df = no_scan.query(selection_query)
    actual_weights = selected_df['actual_weight']

    no_scan_pred, bin_edges = np.histogram(no_scan[var],
                                range=range,
                                bins=bins,
                                weights=actual_weights,
                                )
    no_scan_pred_err = np.histogram(no_scan[var],
                                range=range,
                                bins=bins,
                                weights=actual_weights**2,
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
    '''this function can be wrote in a better way'''
    kde_eff = []
    kde_eff_err = []
    no_scan_eff = []
    for i, m4 in enumerate(exp_analysis_obj.m4_scan):
        kde_eff.append([])
        kde_eff_err.append([])
        no_scan_eff.append([])
        for j, mz in enumerate(exp_analysis_obj.mz_scan):
            if ((exp_analysis_obj.hierarchy == 'heavy') and (m4 >= mz)) or ((exp_analysis_obj.hierarchy == 'light') and (m4 <= mz)):
                kde_eff[-1].append(0)
                kde_eff_err[-1].append(0)
                no_scan_eff[-1].append(0)
                continue
            kde_aux = kde_efficiency(num_selection_query,
                    den_selection_query,
                    m4mz=(m4, mz),
                    exp_analysis_obj=exp_analysis_obj,
                    smoothing_pars=smoothing_pars
                    )
            kde_eff[-1].append(kde_aux[0])
            kde_eff_err[-1].append(kde_aux[1])
            no_scan_aux = no_scan_efficiency(num_selection_query,
                    den_selection_query,
                    m4mz=(m4, mz),
                    exp_analysis_obj=exp_analysis_obj
                    )
            no_scan_eff[-1].append(no_scan_aux[0])

    kde_eff = np.array(kde_eff)
    kde_eff_err = np.array(kde_eff_err)
    no_scan_eff = np.array(no_scan_eff)
    ratio_eff = kde_eff/no_scan_eff
    sigma_ratio_eff = kde_eff_err/no_scan_eff

    xcenters = exp_analysis_obj.m4_scan
    ycenters = exp_analysis_obj.mz_scan

    divnorm = colors.TwoSlopeNorm(vcenter=1, vmin=0, vmax=2)
    plt.pcolormesh(ratio_eff.T, cmap='BrBG', norm=divnorm)
    plt.xticks(ticks=np.arange(0.5, len(xcenters)),
            labels=xcenters)
    plt.yticks(ticks=np.arange(0.5, len(ycenters)),
            labels=ycenters)
    for i in range(len(xcenters)):
        for j in range(len(ycenters)):
            this_value = ratio_eff[i,j]
            if np.isnan(this_value):
                continue
            text = plt.text(i + 0.5, j + 0.5, f"{ratio_eff[i,j]:.3g}\n$\pm${sigma_ratio_eff[i,j]:.2g}",
                            ha="center", va="center", color="k")
    plt.xlabel(r'$m_4$ [GeV]')
    plt.ylabel(r'$m_Z$ [GeV]')
    plt.colorbar()
    set_plot_title(plt.gca(), num_selection_query, (None, None), exp_analysis_obj, smoothing_pars)





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

