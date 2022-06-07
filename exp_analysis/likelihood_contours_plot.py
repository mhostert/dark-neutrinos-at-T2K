import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib import patches
from scipy import interpolate
import scipy.stats

from parameters_dict import likelihood_levels_2d, physics_parameters
from analyses_dict import analyses
from exp_analysis_class import compute_likelihood_from_retrieved

from other_limits.Nlimits import *
from other_limits.DPlimits import *
from other_limits.DPlimits import semi_visible_DP
from other_limits.DPlimits import visible

from math import floor, log10

def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman(f):
    return f/10**fexp(f)

fsize = 12   
def set_plot_style():
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['computer modern roman', 'serif']
    rcParams['figure.figsize'] = (1.2*3.7,1.2*2.3617)
    rcParams['hatch.linewidth'] = 0.3
    rcParams['axes.labelsize'] = fsize
    rcParams['xtick.labelsize'] = fsize
    rcParams['ytick.labelsize'] = fsize
    rcParams['axes.titlesize'] = 10
    rcParams['legend.frameon'] = False
    rcParams['legend.fontsize'] = 0.8*fsize
    rcParams['legend.loc'] = 'best'

def set_canvas_basic():
    fig = plt.figure()
    axes_form = [0.17,0.17,0.79,0.74]
    ax = fig.add_axes(axes_form)
    
    return ax

def set_canvas(plot_type):
    ax = set_canvas_basic()
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_title('Preliminary', loc='right', style='italic')
    
    if plot_type == 'mz_epsilon':
        ax.set_xlabel(r'$m_{Z^{\prime}}$ [GeV]')
        ax.set_ylabel(r'$\varepsilon$')
    elif plot_type == 'm4_Umu4_2':
        ax.set_xlabel(r'$m_{N}$ [GeV]')
        ax.set_ylabel(r'$|V_{\mu N}|^2$')
    return ax

def set_plot_title(ax=None, m4=None, mz=None, alpha_dark=None, epsilon=None, Umu4_2=None, Ud4_2=None, external_ctau=None):
    if ax is None:
        ax = plt.gca()
    
    title_string_elements = []
    if m4 is not None:
        title_string_elements.append(f'$m_N = {m4}$ GeV')
    if mz is not None:
        title_string_elements.append(f'$m_{{Z^{{\prime}}}} = {mz}$ GeV')
    if alpha_dark is not None:
        title_string_elements.append(f'$\\alpha_D = {alpha_dark}$')
    if epsilon is not None:
        if epsilon > 1e-2:
            title_string_elements.append(f'$\\varepsilon = {epsilon:.2g}$')
        else:
            title_string_elements.append(f'$\\varepsilon = {fman(epsilon):.2g}\\times 10^{{{fexp(epsilon)}}}$')
    if Umu4_2 is not None:
        title_string_elements.append(f'$|V_{{\mu N}}|^2 = {fman(Umu4_2):.2g}\\times 10^{{{fexp(Umu4_2)}}}$')
    if Ud4_2 is not None:
        title_string_elements.append(f'$|V_{{N}}|^2 = {Ud4_2}$')
    
    if len(title_string_elements) > 4:
        title_string_elements[int(len(title_string_elements)/2)] = ('\n' + title_string_elements[int(len(title_string_elements)/2)])
    
    if ax is False:
        return ', '.join(title_string_elements)
    else:
        ax.set_title(', '.join(title_string_elements), loc='left')
    
def compute_likes(retrieved, my_exp_analyses, hierarchy, D_or_M, analysis_names, analyses=analyses):
    likes = {}
    mus = {}
    sigma2s = {}
    for analysis_name in analysis_names:
        likes[analysis_name] = 0
        mus[analysis_name] = 0
        sigma2s[analysis_name] = 0
        for nu_mode in analyses[analysis_name].keys():
            print(analysis_name, nu_mode)
            out = compute_likelihood_from_retrieved(retrieved[nu_mode], 
                                              my_exp_analyses[f'{hierarchy}_{D_or_M}_{nu_mode}'], 
                                              analyses[analysis_name][nu_mode], 
                                              like_normalized=True)
            likes[analysis_name] += out[0]
            mus[analysis_name] += out[1]
            sigma2s[analysis_name] += out[2]
            del out
            gc.collect()
        likes[analysis_name] -= likes[analysis_name].min()
    return likes, mus, sigma2s

def combine_likes(likes):
    total_like = 0
    for like in likes:
        total_like += like
    total_like -= like.min()
    return total_like

def basic_contour_plot(case_vars,
                       retrieved, 
                       likes,
                       analysis_names,
                       hierarchy='heavy',
                       D_or_M='dirac',
                       save_name=None,
                       save_folder=None,
                       ax=None,
                       legend_loc='best',
                       colors=['deepskyblue', 'blue', 'navy'],
                       fill=None,
                       poster_setting=False,
                       linestyles=['-', '-', '-'], 
                       legend_outside=False,
                       levels=[likelihood_levels_2d[0.9], np.inf]):
    contours = {}
    if ax is None:
        ax = set_canvas(f'{case_vars[0]}_{case_vars[1]}')
        ax.set_xlim(retrieved['FHC']['pars'][case_vars[0]][0], retrieved['FHC']['pars'][case_vars[0]][-1])
        ax.set_ylim(retrieved['FHC']['pars'][case_vars[1]][0], retrieved['FHC']['pars'][case_vars[1]][-1])
    for i, analysis_name in enumerate(analysis_names):
        contours[analysis_name] = ax.contour(retrieved['FHC']['pars'][case_vars[0]], 
                                             retrieved['FHC']['pars'][case_vars[1]], 
                                             likes[analysis_name].T, 
                                             levels=levels, 
                                             colors=[colors[i]], linestyles=[linestyles[i]])
        if fill is not None:
            if fill[i]:
                ax.contourf(retrieved['FHC']['pars'][case_vars[0]], 
                                                 retrieved['FHC']['pars'][case_vars[1]], 
                                                 likes[analysis_name].T, 
                                                 levels=levels, 
                                                 colors=[colors[i]], alpha=0.2)
    ax.loglog()
    
    if not poster_setting:
        ax.plot(physics_parameters[hierarchy]['bp'][case_vars[0]],
                physics_parameters[hierarchy]['bp'][case_vars[1]],
                'r*',
                label='Benchmark\n point')
        set_plot_title(ax, **{key:value for (key,value) in retrieved['FHC']['pars'].items() if key not in case_vars})
    else:
        ax.set_title('Preliminary', loc='right', style='italic', fontsize=fsize*0.9)
    
    handles, labels = ax.get_legend_handles_labels()
    handles += [cntr.legend_elements()[0][0] for cntr in contours.values()]
    labels += contours.keys()
    if not legend_outside:
        ax.legend(handles,
                  labels,
                   frameon=False,
                   loc=legend_loc)
    else:
        ax.legend(handles,
                  labels,
                  frameon=False,
                  bbox_to_anchor=(1.8, 0.2))
    
    nticks = 10
    maj_loc = ticker.LogLocator(numticks=nticks)
    min_loc = ticker.LogLocator(subs='all', numticks=nticks)
    ax.yaxis.set_major_locator(maj_loc)
    ax.yaxis.set_minor_locator(min_loc)
    if save_name is not None:
        if not poster_setting:
            plt.savefig(save_folder + f'{hierarchy}_{D_or_M}_{case_vars[0]}_{case_vars[1]}_{save_name}.pdf')
        else:
            plt.savefig(save_folder + f'{hierarchy}_{D_or_M}_{case_vars[0]}_{case_vars[1]}_{save_name}.png', dpi=500, transparent=True)
        
def mz_epsilon_heavy_plot(ax, m4, mz_ticks, poster_setting=False):
    FNAL_run_combined = gminus2.weighted_average(gminus2.DELTA_FNAL, gminus2.DELTA_BNL)

    energy, one_over_alpha = np.loadtxt("./other_limits/DPlimits/digitized/alphaQED/alpha_QED_running_posQ2.dat", unpack = True)
    one_over_alpha_ew = interpolate.interp1d(energy, one_over_alpha, kind="linear")

    FACTOR = 1/2/np.pi/one_over_alpha_ew(gminus2.M_MU)
    
    gminus2_sigmas = [2.]
    gminus2_colors = ['dodgerblue']

    semi_visible_DP.plot_constraints(ax, mz_ticks[0], mz_ticks[-1], separated=False, poster_setting=poster_setting)

    gminus2.compute_and_plot_gminus2_region(
        ax = ax,
        mz = mz_ticks,
        delta_amu = FNAL_run_combined[0],
        error = FNAL_run_combined[1],
        factor = FACTOR,
        sigmas = gminus2_sigmas,
        colors = gminus2_colors,
    )

    ax.annotate(r'$(g-2)_\mu$', xy=(2e-1,4.5e-3), rotation=7, fontsize=0.7*fsize, color='darkblue')

    # miniboone ROI
    plt.fill_between([1, 2], [2e-3, 2e-3], [2.5e-2, 2.5e-2], color='green', alpha=0.4)
    ax.annotate('MiniBooNE\nROI', xy=(2.3, 2e-3), fontsize=0.7*fsize, color='green')
    
    ax.set_xlim(mz_ticks[0], mz_ticks[-1])
    
def m4_Umu4_2_heavy_plot(ax, m4_ticks):
    usqr_bound_inv = umu4.USQR_inv(m4_ticks)

    ax.fill_between(m4_ticks, usqr_bound_inv, np.ones(np.size(m4_ticks)), 
                fc='lightgrey', ec='None', lw =0.0, alpha=0.95)
    ax.annotate('Model\nindependent\nconstraints', xy=(0.0012, 3e-4), rotation=0, fontsize=0.7*fsize, color='black')

    plt.fill_between([0.100, 0.300], [1e-7, 1e-7], [1e-5, 1e-5], color='green', alpha=0.4)
    ax.annotate('MiniBooNE\nROI', xy=(0.4, 1e-7), fontsize=0.7*fsize, color='green')
    ax.set_xlim(m4_ticks[0], m4_ticks[-1])
    ax.set_ylim(1e-10, 1e-2)

def plot_band(ax, mheavy, y, nevents, chi2, color_l, color, ls=1, alpha=0.1, label='', Nint=100):
    mheavyl = np.log10(mheavy)
    yl = np.log10(y)
    xi = np.logspace(mheavyl.min(), mheavyl.max(), Nint)
    yi = np.logspace(yl.min(), yl.max(), Nint)

    zi = scipy.interpolate.griddata((mheavy, y), chi2,\
                                    (xi[None,:], yi[:,None]),\
                                    method='linear', fill_value="Nan", rescale=True)

    Contour = ax.contour(xi, yi, zi, [4.61], colors=['None'], linewidths=0, alpha=alpha)

    l1 = Contour.collections[0].get_paths()[0].vertices  # grab the 1st path

    xint = np.logspace(-1.44, 0, 100)
    f1 = np.interp(xint, l1[:,0], l1[:,1])

    l, = ax.plot(xint, f1, color=color_l, ls='-', lw=1.2, zorder=5, label=label)
    ax.fill_between(xint, f1,np.ones(np.size(f1)), color=color, linewidth=0.0, alpha=alpha)
    return l
    
def m4_Umu4_2_light_plot(ax):
    x,y = np.loadtxt("../digitized/Pedros_paper/experimental_constraints.dat", unpack=True)
    ax.fill_between(x, np.sqrt(y), np.ones(np.size(y)), hatch='\\\\\\\\\\\\\\\\',facecolor='None', alpha=0.5, lw=0.0)

    ALPHA_FIT = 0.4
    xu,yu = np.loadtxt("../digitized/Pedro_v3/upper_3_sigma.dat", unpack=True)
    xl,yl = np.loadtxt("../digitized/Pedro_v3/low_3_sigma.dat", unpack=True)
    x = np.logspace(np.log10(0.042), np.log10(0.68), 100)
    up = np.interp(x,xu,yu)
    low = np.interp(x,xl,yl)
    ax.fill_between(x, low,up, facecolor='green', lw=0.0, alpha=ALPHA_FIT)
    ax.annotate(r'Best fit $3 \, \sigma$', xy=(0.06, 1.3e-9), fontsize=0.7*fsize, color='green')
    # ax.fill_between(x, low,up, edgecolor='black', facecolor="None", lw=0.1, linestyle = '-', alpha=ALPHA_FIT)

    # ax.fill_between(x, low,up, facecolor='orange', lw=0.0, alpha=ALPHA_FIT, label=r'Best fit $3 \, \sigma$')
    # ax.fill_between(x, low,up, edgecolor='black', facecolor="None", lw=0.1, linestyle = '-', alpha=ALPHA_FIT)
    
    xu,yu = np.loadtxt("../digitized/Pedro_v3/upper_1_sigma.dat", unpack=True)
    xl,yl = np.loadtxt("../digitized/Pedro_v3/low_1_sigma.dat", unpack=True)
    x = np.logspace(np.log10(0.068), np.log10(0.21), 100)
    up = np.interp(x,xu,yu)
    low = np.interp(x,xl,yl)
    ax.fill_between(x, low,up, facecolor='yellow', lw=0.7, alpha=ALPHA_FIT)
    ax.annotate(r'Best fit $1 \, \sigma$', xy=(0.06, 3.4e-9), fontsize=0.7*fsize, color='yellow')
    # ax.fill_between(x, low,up, edgecolor='black', facecolor="None", lw=0.1, linestyle = '-', alpha=ALPHA_FIT)
    
    # ax.fill_between(x, low,up, facecolor='#EFFF00', lw=0.7, alpha=ALPHA_FIT, label=r'Best fit $1 \, \sigma$')
    # ax.fill_between(x, low,up, edgecolor='black', facecolor="None", lw=0.1, linestyle = '-', alpha=ALPHA_FIT)
    
    sensitivity_LE = np.load("../digitized/minerva_charm_limits/sensitivity_LE_vAugust2019.npy")
    sensitivity_ME = np.load("../digitized/minerva_charm_limits/sensitivity_ME_vAugust2019.npy")
    sensitivity_CH = np.load("../digitized/minerva_charm_limits/sensitivity_CH_vAugust2019.npy")
    sensitivity_CH_bar = np.load("../digitized/minerva_charm_limits/sensitivity_CH_bar_vAugust2019.npy")

    # Plot fit
    mheavy_LE, Umu42_LE, nevents_LE, chi2_LE = sensitivity_LE
    mheavy_ME, Umu42_ME, nevents_ME, chi2_ME = sensitivity_ME
    mheavy_CH, Umu42_CH, nevents_CH, chi2_CH = sensitivity_CH
    mheavy_CH, Umu42_CH, nevents_CH_bar, chi2_CH_bar = sensitivity_CH_bar

    chi2_CH += chi2_CH_bar

    CHARM_COLOR = "dodgerblue"
    CHARM_COLOR_L = "dodgerblue"

    # MINERVA_COLOR = "green"
    # MINERVA_COLOR_L = "darkgreen"

    ME_COLOR = "navy"
    ME_COLOR_L = "navy"

    # shouldn't this be the sum of nevents_CH + nevents_CH_bar
    l1 = plot_band(ax, mheavy_CH, Umu42_CH, nevents_CH, chi2_CH, CHARM_COLOR_L, CHARM_COLOR)
    ax.annotate(r'CHARM-II', xy=(0.4, 8e-9), fontsize=0.7*fsize, color=CHARM_COLOR_L)
    # plot_band(ax, mheavy_LE, Umu42_LE, nevents_LE, chi2_LE, label, MINERVA_COLOR_L, MINERVA_COLOR)
    l3 = plot_band(ax, mheavy_ME, Umu42_ME, nevents_ME, chi2_ME, ME_COLOR_L, ME_COLOR)
    ax.annotate(r'MINER$\nu$A ME', xy=(0.43, 8e-8), fontsize=0.7*fsize, color=ME_COLOR_L)
    # ax.legend(frameon=False, handles=[l1,l3], fontsize=0.8*fsize, loc='lower right')
    # ax.legend(frameon=False, loc='lower right')
    ########################################
    # Other model-indepenedent limits

    # ax.plot(x, y, color='black', lw=0.5, hatch='///')

def mz_epsilon_light_plot(ax):
    visible.light_dark_photon_limits(ax)

def plot_data_from_analysis(ax, analysis, plot_data=True):
    if analysis['var'] is not None:
        binning = analysis['binning']
    else:
        binning = np.array([-1, 1])

    bin_centers = np.atleast_1d((binning[:-1] + binning[1:])/2)
    bin_width = binning[1] - binning[0]

    analysis['mc'] = np.atleast_1d(analysis['mc'])
    ax.bar(bin_centers, 
           analysis['mc'], 
            width=bin_width,
            alpha=0.2, color='C0', label='SM expectation')
    
    for i, (m, v) in enumerate(zip(bin_centers, analysis['mc'])):
        e = v * analysis['syst']
        if i == 0:
            ax.add_patch(
                patches.Rectangle(
                    (m - bin_width/2, v - e),
                    bin_width,
                    2 * e,
                    hatch="\\\\\\\\\\",
                    fill=False,
                    linewidth=0,
                    alpha=0.4,
                    label='systematics',
                )
            )
        else:
            ax.add_patch(
                patches.Rectangle(
                    (m - bin_width/2, v - e),
                    bin_width,
                    2 * e,
                    hatch="\\\\\\\\\\",
                    fill=False,
                    linewidth=0,
                    alpha=0.4,
                )
            )
    
    if plot_data:
        analysis['data'] = np.atleast_1d(analysis['data'])
        ax.errorbar(bin_centers, analysis['data'], yerr=np.sqrt(analysis['data']), fmt='k.', label='data')
    
    ax.set_ylabel(r'Number of entries')