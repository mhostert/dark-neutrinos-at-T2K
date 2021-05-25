import numpy as np
from scipy.integrate import dblquad

from parameters_dict import *
from const import *

def ctau_heavy(m4, mz, Valpha4_alphaepsilon2, D_or_M):
    return get_decay_rate_in_cm(gamma_heavy(m4, mz, Valpha4_alphaepsilon2, D_or_M))

def ctau_light(m4, mz, Valpha4, D_or_M):
    return get_decay_rate_in_cm(gamma_N_light(m4, mz, Valpha4, D_or_M))

def gamma_heavy(m4, mz, Valpha4_alphaepsilon2, D_or_M):
    # to accept floats
    m4 = np.asarray(m4)
    mz = np.asarray(mz)

    '''there is a cancellation for small r that holds up to 4th order, so I avoid instability by expanding when r is small'''
    r = ((m4/mz)**2)
    gamma = Valpha4_alphaepsilon2/12.0/np.pi/r**2 * m4
    # avoiding evaluation of bad expression
    mask = (r>0.01)
    piece = np.empty_like(r)
    piece[mask] = (6*(r[mask] - r[mask]**2/2.0 - np.log((1.0/(1.0-r[mask]))**(1 - r[mask])) )- r[mask]**3)
    piece[~mask] = r[~mask]**4/2

    gamma *= piece
    if D_or_M == 'dirac':
        gamma /= 2
    return gamma

def gamma_N_light(m4, mz, Valpha4, D_or_M):
    gamma = 1/2 *m4**3/mz**2 * (1-mz**2/m4**2)**2 * (0.5+mz**2/m4**2)
    if D_or_M == 'dirac':
        gamma /= 2
    gamma *= Valpha4
    return gamma

def gamma_Zprime_light(m4, mz, epsilon2, m_ell=m_e):
    gamma = (2*np.pi*2*np.sqrt(-4*(m_ell*m_ell) + mz*mz)*(5*(m_ell*m_ell) + 2*(mz*mz))*np.pi)/(24.*(mz*mz)*(np.pi*np.pi))
    gamma *= alphaQED
    gamma *= epsilon2
    return gamma

def gamma_light(m4, mz, Valpha4_alphaepsilon2, D_or_M, m_ell=m_e):
    '''return the product of GammaN * GammaZprime'''
    return gamma_N_light(m4, mz, Valpha4_alphaepsilon2/alphaQED, D_or_M) * gamma_Zprime_light(m4, mz, 1, m_ell)
    
def gamma_general(m4, mz, Valpha4alphaepsilon2, D_or_M):
    heavy = (m4 < mz)
    gamma = np.empty_like(m4)
    gamma[heavy] = gamma_heavy(m4[heavy], mz[heavy], Valpha4alphaepsilon2, D_or_M)
    gamma[~heavy] = gamma_light(m4[~heavy], mz[~heavy], Valpha4alphaepsilon2, D_or_M)
    return gamma

def gamma_heavy_contact(m4, mz, Valpha4_alphaepsilon2, D_or_M):
    gamma = Valpha4_alphaepsilon2/(24 * np.pi) * m4**5/mz**4*np.heaviside(mz - m4,0)
    if D_or_M == 'dirac':
        gamma /= 2
    return gamma

def gamma_heavy_contact_integrated(m4_s, mz_s, Valpha4_alphaepsilon2, normalised=True):
    aux = Valpha4_alphaepsilon2/(24 * np.pi) * (1/6) * (1/(-3))
    aux *= (m4_s[1]**6 - m4_s[0]**6)
    aux *= (mz_s[1]**(-3) - mz_s[0]**(-3))
    if normalised:
        aux /= ((m4_s[1] - m4_s[0])*(mz_s[1] - mz_s[0]))
    return aux

def gamma_heavy_integrated(m4_s, mz_s, Valpha4_alphaepsilon2, normalised=True):
    aux, _ = dblquad(gamma_heavy,
                    mz_s[1], mz_s[0],
                    m4_s[1], m4_s[0],
                    args=[Valpha4_alphaepsilon2],
                    epsrel=1e-8)
    if normalised:
        aux /= ((m4_s[1] - m4_s[0])*(mz_s[1] - mz_s[0]))
    return aux