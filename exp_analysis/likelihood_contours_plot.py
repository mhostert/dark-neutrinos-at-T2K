import gc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import interpolate
import scipy.stats

from parameters_dict import likelihood_levels_2d
from analyses_dict import analyses
from exp_analysis_class import compute_likelihood_from_retrieved

from other_limits.Nlimits import *
from other_limits.DPlimits import *

# from scipy.integrate import quad


# from matplotlib import rc, rcParams
# from matplotlib.pyplot import *
# from matplotlib.legend_handler import HandlerLine2D
# import matplotlib.colors as colors
# import scipy.ndimage as ndimage


fsize = 12
def set_plot_style():
    # rcParams['text.usetex'] = True
    rcParams['text.usetex'] = False
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman', 'serif']
    rcParams['figure.figsize'] = (1.2*3.7,1.2*2.3617)
    # rcParams['lines.linewidth'] = 1.0
    rcParams['hatch.linewidth'] = 0.3
    # rcParams['axes.linewidth'] = 0.4
    rcParams['axes.labelsize'] = fsize
    # rcParams['xtick.direction'] = 'in'
    # rcParams['xtick.major.width'] = 0.4
    # rcParams['xtick.minor.width'] = 0.4
    rcParams['xtick.labelsize'] = fsize
    # rcParams['ytick.direction'] = 'in'
    # rcParams['ytick.major.width'] = 0.4
    # rcParams['ytick.minor.width'] = 0.4
    rcParams['ytick.labelsize'] = fsize
    rcParams['legend.frameon'] = False
    rcParams['legend.fontsize'] = 0.8*fsize
    rcParams['legend.loc'] = 'lower right'
    # rcParams["text.latex.preamble"] = r'''
    #     \usepackage{amsmath,amssymb,amsthm}
    #     \usepackage{siunitx}
    # '''

def set_canvas(plot_type):
    fig = plt.figure()
    axes_form = [0.14,0.15,0.82,0.76]
    ax = fig.add_axes(axes_form)
    # ax.set_xlim(X_MIN, X_MAX)
    # ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Preliminary', loc='right', style='italic')
    
    if plot_type == 'mz_epsilon':
        ax.set_xlabel(r'$m_{Z^\prime}$ [GeV]')
        ax.set_ylabel(r'$\varepsilon$')
    elif plot_type == 'm4_Umu4_2':
        ax.set_xlabel(r'$m_{N}$ [GeV]')
        ax.set_ylabel(r'$|U_{\mu N}|^2$')
    return ax

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
                       save_name=None,
                       save_folder=None,
                       ax=None,
                       legend_loc='best',
                       colors=['deepskyblue', 'blue', 'navy'],
                       linestyles=['-', '-', '-'], levels=[likelihood_levels_2d[0.9]]):
    contours = {}
    if ax is None:
        ax = set_canvas(f'{case_vars[0]}_{case_vars[1]}')
    for i, analysis_name in enumerate(analysis_names):
        contours[analysis_name] = ax.contour(retrieved['FHC']['pars'][case_vars[0]], 
                                             retrieved['FHC']['pars'][case_vars[1]], 
                                             likes[analysis_name].T, 
                                             levels=levels, 
                                             colors=[colors[i]], linestyles=[linestyles[i]])
    ax.loglog()
    
    handles, labels = ax.get_legend_handles_labels()
    handles += [cntr.legend_elements()[0][0] for cntr in contours.values()]
    labels += contours.keys()
    
    ax.legend(handles,
              labels,
               frameon=False,
               loc=legend_loc)
    if save_name is not None:
        plt.savefig(save_folder + f'{case_vars[0]}_{case_vars[1]}_{save_name}.pdf', bbox_inches='tight')

def mz_epsilon_heavy_plot(ax, m4, mz_ticks):
    FNAL_run_combined = gminus2.weighted_average(gminus2.DELTA_FNAL, gminus2.DELTA_BNL)

    energy, one_over_alpha = np.loadtxt("./other_limits/DPlimits/digitized/alphaQED/alpha_QED_running_posQ2.dat", unpack = True)
    one_over_alpha_ew = interpolate.interp1d(energy, one_over_alpha, kind="linear")

    FACTOR = 1/2/np.pi/one_over_alpha_ew(gminus2.M_MU)
    
    gminus2_sigmas = [2.]
    gminus2_colors = ['dodgerblue']

    semi_visible_DP.plot_constraints(ax, mz_ticks[0], mz_ticks[-1], separated=False)

    gminus2.compute_and_plot_gminus2_region(
        ax = ax,
        mz = mz_ticks,
        delta_amu = FNAL_run_combined[0],
        error = FNAL_run_combined[1],
        factor = FACTOR,
        sigmas = gminus2_sigmas,
        colors = gminus2_colors
    )

#     ax.axvline(x=m4, color='black', lw=1)
#     ax.annotate('', xy=(2e-2, 1.2e-4), xytext=(m4, 1.2e-4), 
#                 arrowprops=dict(arrowstyle="-|>", mutation_scale=7, color='black', lw=1))

#     ax.annotate(r"$m_{4} > m_{Z'}$"+'\nN short lived\nno constraint', 
#                 xy=(9e-2, 3e-5), 
#                 fontsize=0.7*fsize, 
#                 color='black', 
#                 horizontalalignment='right')
    ax.annotate(r'$(g-2)_\mu$', xy=(2e-2,1.6e-3), rotation=4, fontsize=0.7*fsize, color='darkblue')

    # miniboone ROI
    plt.fill_between([1, 2], [2e-3, 2e-3], [2.5e-2, 2.5e-2], color='green', alpha=0.4)
    ax.annotate('MiniBooNE\nROI', xy=(2.3, 5e-3), fontsize=0.7*fsize, color='green')

    ax.set_title('Preliminary', loc='right', style='italic')
    ax.set_xlim(mz_ticks[0], mz_ticks[-1])
    
def m4_Umu4_2_heavy_plot(ax, m4_ticks):
    # usqr_bound = umu4.USQR(m4_ticks)
    usqr_bound_inv = umu4.USQR_inv(m4_ticks)

    ##############################################
    # Constraints on U\alpha4^2
    # Minimal HNL -- no Zprime and all that
    # ax.plot(MN, usqr_bound, color='navy', )
    # ax.fill_between(MN, usqr_bound, np.ones(np.size(MN)), 
    #             fc='dodgerblue', ec='None', lw =0.0, alpha=0.5, label=r'all bounds')

    # most model independent bounds
    # ax.plot(MN, usqr_bound_inv, color='navy', lw=1)
    ax.fill_between(m4_ticks, usqr_bound_inv, np.ones(np.size(m4_ticks)), 
                fc='lightgrey', ec='None', lw =0.0, alpha=0.95)
    ax.annotate('Model\nindependent\nconstraints', xy=(0.0012, 0.02), rotation=0, fontsize=0.7*fsize, color='black')

    # MiniBooNE ROI
    plt.fill_between([0.100, 0.300], [1e-7, 1e-7], [1e-5, 1e-5], color='green', alpha=0.4)
    ax.annotate('MiniBooNE\nROI', xy=(0.4, 4e-7), fontsize=0.7*fsize, color='green')

    # ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])

def plot_band(ax, mheavy, y, nevents, chi2, color_l, color, ls=1, alpha=0.1, label='', Nint=100):
    mheavyl = np.log10(mheavy)
    yl = np.log10(y)
    xi = np.logspace(mheavyl.min(), mheavyl.max(), Nint)
    yi = np.logspace(yl.min(), yl.max(), Nint)

    zi = scipy.interpolate.griddata((mheavy, y), chi2,\
                                    (xi[None,:], yi[:,None]),\
                                    method='linear', fill_value="Nan", rescale=True)

    # zi_g = scipy.ndimage.filters.gaussian_filter(zi, 0.8, mode='nearest', order = 0, cval=0)
    # lw = 1.3
    Contour = ax.contour(xi, yi, zi, [4.61], colors=['None'], linewidths=0, alpha=alpha)

    l1 = Contour.collections[0].get_paths()[0].vertices  # grab the 1st path

    xint = np.logspace(-1.44, 0, 100)
    f1 = np.interp(xint, l1[:,0], l1[:,1])

    l, = ax.plot(xint, f1, color=color_l, ls='-', lw=1.2, zorder=5, label=label)
    ax.fill_between(xint, f1,np.ones(np.size(f1)), color=color, linewidth=0.0, alpha=alpha)
    return l
    
def m4_Umu4_2_light_plot(ax):
    # model independent limits
    # ax.annotate(r"Excluded", xy=(0.12,0.04), fontsize=fsize, color = 'black' )
    x,y = np.loadtxt("../digitized/Pedros_paper/experimental_constraints.dat", unpack=True)
    ax.fill_between(x, np.sqrt(y), np.ones(np.size(y)), hatch='\\\\\\\\\\\\\\\\',facecolor='None', alpha=0.5, lw=0.0)

    # fsize = 10
    # # S_GAUSS = 0.1
    # rc('text', usetex=True)
    # params={'axes.labelsize':fsize,
    #         'xtick.labelsize':fsize,
    #         'ytick.labelsize':fsize,
    #         'figure.figsize':(6, 5)}
    #         # 'figure.figsize':(3.39,1.2*2.3617)}
    # rc('font', **{'family':'serif', 'serif': ['computer modern roman']})
    # rcParams.update(params)
    # axes_form  = [0.18,0.16,0.77,0.75]

    # fig = plt.figure()
    # ax = fig.add_axes(axes_form)

    

    # proxy2 = plt.Rectangle((0,0), 1, 1, fc = "orange", ec=None, alpha=ALPHA_FIT, lw=0.7) 
    # proxy1 = plt.Rectangle((0,0), 1, 1, fc = "#EFFF00", ec=None, alpha=ALPHA_FIT, lw=0.7) 

    # leg = ax.legend([proxy1,proxy2], [r"$1 \sigma$", r"$3 \sigma$"], fontsize=fsize*0.85, frameon=False, loc=(0.078, 0.470), ncol=1)
    # ax.add_artist(leg);
    # ax.annotate('MiniBooNE', xy=(0.038,0.0016), fontsize=0.9*fsize, color = 'black' )
    # ax.annotate('energy fit', xy=(0.043,0.001), fontsize=0.9*fsize, color = 'black' )

    # plt.setp(leg.get_title(),fontsize=fsize*0.85)
    ALPHA_FIT = 0.9
    xu,yu = np.loadtxt("../digitized/Pedro_v3/upper_3_sigma.dat", unpack=True)
    xl,yl = np.loadtxt("../digitized/Pedro_v3/low_3_sigma.dat", unpack=True)
    x = np.logspace(np.log10(0.042), np.log10(0.68), 100)
    up = np.interp(x,xu,yu)
    low = np.interp(x,xl,yl)
    ax.fill_between(x, low,up, facecolor='orange', lw=0.0, alpha=ALPHA_FIT, label=r'$3 \, \sigma$')
    ax.fill_between(x, low,up, edgecolor='black', facecolor="None", lw=0.1, linestyle = '-', alpha=ALPHA_FIT)

    xu,yu = np.loadtxt("../digitized/Pedro_v3/upper_1_sigma.dat", unpack=True)
    xl,yl = np.loadtxt("../digitized/Pedro_v3/low_1_sigma.dat", unpack=True)
    x = np.logspace(np.log10(0.068), np.log10(0.21), 100)
    up = np.interp(x,xu,yu)
    low = np.interp(x,xl,yl)
    ax.fill_between(x, low,up, facecolor='#EFFF00', lw=0.7, alpha=ALPHA_FIT, label=r'$1 \, \sigma$')
    ax.fill_between(x, low,up, edgecolor='black', facecolor="None", lw=0.1, linestyle = '-', alpha=ALPHA_FIT)

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

    MINERVA_COLOR = "green"
    MINERVA_COLOR_L = "darkgreen"

    ME_COLOR = "firebrick"
    ME_COLOR_L = "darkred"

    # shouldn't this be the sum of nevents_CH + nevents_CH_bar
    l1 = plot_band(ax, mheavy_CH, Umu42_CH, nevents_CH, chi2_CH, CHARM_COLOR_L, CHARM_COLOR, label='CHARM-II')
    # plot_band(ax, mheavy_LE, Umu42_LE, nevents_LE, chi2_LE, label, MINERVA_COLOR_L, MINERVA_COLOR)
    l3 = plot_band(ax, mheavy_ME, Umu42_ME, nevents_ME, chi2_ME, ME_COLOR_L, ME_COLOR, label='MINERvA ME')

    # ax.legend(frameon=False, handles=[l1,l3], fontsize=0.8*fsize, loc='lower right')
    # ax.legend(frameon=False, loc='lower right')
    ########################################
    # Other model-indepenedent limits

    # ax.plot(x, y, color='black', lw=0.5, hatch='///')

