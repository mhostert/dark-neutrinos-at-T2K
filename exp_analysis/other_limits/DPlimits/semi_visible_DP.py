import numpy as np
import scipy.interpolate as interpolate
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import os
PATH = os.path.dirname(os.path.abspath(__file__))+'/digitized'

def fpath(*paths):
    return os.path.join(PATH, *paths)

INFTY = 1e100

def bound(filename, xmin, xmax, npoints = 1000):
    ''' get x, y data from digitized data and interpolate, assuming they are logarithmically distributed '''
    xp, yp = np.genfromtxt(filename, unpack=True)
    y = interpolate.interp1d(np.log10(xp), np.log10(yp), bounds_error = False, fill_value = np.log10(INFTY)) 
    x = np.linspace(np.log10(xmin), np.log10(xmax), npoints)
    return [np.power(10, x), np.power(10, y(x))]

def plot_constraints(ax, xmin, xmax, separated=True, poster_setting=False):
    # GENERAL BOUNDS
    lw = 0.5
    dashes = (6, 0.1)
    alpha = 0.95
    color_line = 'black'
    color_fill = 'lightgrey'
    fsize = 12
    
    x, y = bound(fpath('BESIII', 'g-2e.dat'), xmin = xmin , xmax = xmax)
    if separated:
        ax.plot(x, y, color = color_line, lw = lw, dashes = dashes)
    ax.fill_between(x, y, np.ones(np.size(y)), color = color_fill, lw = 0.0, alpha = alpha)
    # ax.fill_between(x, y, np.ones(np.size(y)), facecolor = color_fill, edgecolor = color_line, lw = lw, alpha = alpha)
    if separated:
        ax.annotate(r'$(g-2)_e$', xy=(1.4e-3, 2e-4), rotation=28, fontsize=0.7*fsize, color=color_line)

    x, y = bound(fpath('BESIII', 'NA62.dat'), xmin = xmin , xmax = xmax)
    if separated:
        ax.plot(x,y,color = color_line, lw = lw, dashes = dashes)
    ax.fill_between(x, y, np.ones(np.size(y)), color = color_fill, lw = 0.0, alpha = alpha)
    # ax.fill_between(x, y, np.ones(np.size(y)), facecolor = color_fill, edgecolor = color_line, lw = lw, alpha = alpha)
    if separated:
        ax.annotate(r'NA62', xy=(3.3e-2, 5e-3), rotation=90, fontsize=0.7*fsize, color=color_line)

    x, y = bound(fpath('BESIII', 'E949.dat'), xmin = xmin , xmax = xmax)
    if separated:
        ax.plot(x, y, color = color_line, lw = lw, dashes = dashes)
    ax.fill_between(x, y, np.ones(np.size(y)), color = color_fill, lw = 0.0, alpha = alpha)
    # ax.fill_between(x, y, np.ones(np.size(y)), facecolor = color_fill, edgecolor = color_line, lw = lw, alpha = alpha)
    if separated:
        ax.annotate(r'E949', xy=(1.6e-2, 1.1e-2), rotation=90, fontsize=0.7*fsize, color=color_line)

    x, y = bound(fpath('BESIII', 'BaBar.dat'), xmin = xmin , xmax = xmax)
    if separated:
        ax.plot(x, y, color = color_line, lw = lw, dashes = (3,1))
#     ax.fill_between(x, y, np.ones(np.size(y)), facecolor = 'None', edgecolor='black', hatch='\\\\\\', lw = 0.5, alpha = alpha, zorder=-1)
    ax.fill_between(x, y, np.ones(np.size(y)), facecolor = 'None', edgecolor='black', lw = 0.5, alpha = alpha, zorder=-1)
    # ax.fill_between(x, y, np.ones(np.size(y)), facecolor = color_fill, edgecolor = color_line, lw = lw, alpha = alpha)
    # if not poster_setting:
    #     babar_annotation = r'BaBar$^*$'+'\n'+r'$|V_{{ND}}|^2 = 10^{-4}$'
    # else:
    #     babar_annotation = r'BaBar$^*$'
    babar_annotation = r'BaBar$^*$'
    ax.annotate(babar_annotation, xy=(5.5, 0.9e-3), rotation=0, fontsize=0.7*fsize, color=color_line, horizontalalignment='right')

    x, y = bound(fpath('Curtin_et_al', 'LHC_current.dat'), xmin = xmin, xmax = xmax)
    if separated:
        ax.plot(x, y, color = color_line, lw = lw, dashes = dashes)
    ax.fill_between(x, y, np.ones(np.size(y)), color = color_fill, lw = 0.0, alpha = alpha)
    # ax.fill_between(x, y, np.ones(np.size(y)), facecolor = color_fill, edgecolor = color_line, lw = lw, alpha = alpha)
    if separated:
        ax.annotate(r'EWPO', xy=(3e-3, 3.1e-2), rotation=0, fontsize=0.7*fsize, color=color_line)

    x, y = bound(fpath('DISMcKeen', 'DIS_bounds.dat'), xmin = xmin, xmax = xmax)
    if separated:
        ax.plot(x, y, color = color_line, lw = lw, dashes = dashes)
    ax.fill_between(x, y, np.ones(np.size(y)), color = color_fill, lw = 0.0, alpha = alpha)
    if separated:
        ax.annotate(r'DIS', xy=(4, 1.3e-2), rotation=6, fontsize=0.7*fsize, color=color_line)

    if not separated:
        ax.annotate('Other dark\nphoton constraints', xy=(0.15, 0.022), rotation=0, fontsize=0.7*fsize, color=color_line)