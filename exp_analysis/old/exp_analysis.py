import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import interpn
from scipy.integrate import dblquad
import pandas as pd
import matplotlib.pyplot as plt
from . import fourvec 

def unitary_decay_length(df):
    p3dark = np.sqrt(dot3_df(df['pdark'], df['pdark']))
    mdark = inv_mass(df['pdark'])
    betagamma = p3dark/mdark
    gamma = df['pdark', 't']/mdark
    beta = betagamma/gamma
    
    d_decay = np.random.exponential(scale=betagamma) # it's for ctau=1

    df[f'unitary_decay_length', 't'] = d_decay/(2.998e+10 * beta)
    df[f'unitary_decay_length', 'x'] = d_decay*df['pdark', 'x']/p3dark
    df[f'unitary_decay_length', 'y'] = d_decay*df['pdark', 'y']/p3dark
    df[f'unitary_decay_length', 'z'] = d_decay*df['pdark', 'z']/p3dark


def decay_particle(df, ctau):
    p3dark = np.sqrt(dot3_df(df['pdark'], df['pdark']))
    mdark = inv_mass(df['pdark'])
    betagamma = p3dark/mdark
    gamma = df['pdark', 't']/mdark
    beta = betagamma/gamma
    
    d_decay = np.random.exponential(scale=ctau*betagamma) # centimeters

    df[f'decay_point_{ctau}', 't'] = df['int_point', 't'] + d_decay/(2.998e+10 * beta)
    df[f'decay_point_{ctau}', 'x'] = df['int_point', 'x'] + d_decay*df['pdark', 'x']/p3dark
    df[f'decay_point_{ctau}', 'y'] = df['int_point', 'y'] + d_decay*df['pdark', 'y']/p3dark
    df[f'decay_point_{ctau}', 'z'] = df['int_point', 'z'] + d_decay*df['pdark', 'z']/p3dark

# compute useful variables
def compute_analysis_variables(df):
    for comp in ['t','x','y','z']:
        df['pee', comp] = df['plm', comp] + df['plp', comp]
        df['pdark', comp] = df['plm', comp] + df['plp', comp] + df['pnu', comp]
    df['recoil_mass', ''] = inv_mass(df['pHad']).round(6)
    df['ee_mass', ''] = inv_mass(df['pee'])
    df['ee_energy', ''] = df['pee', 't']
    df['ee_costheta', ''] = costheta(df['plm'], df['plp'])
    df['nu_dark_beam_costheta', ''] = df['pdark', 'z']/np.sqrt(dot3_df(df['pdark'], df['pdark']))
    df['ee_beam_costheta', ''] = df['pee', 'z']/np.sqrt(dot3_df(df['pee'], df['pee']))
    df['ee_momentum', ''] = np.sqrt(dot3_df(df['pee'], df['pee']))
    df['experimental_t', ''] = (df['plm','t'] - df['plm','z'] + df['plp','t'] - df['plp','z'])**2 +\
                                df['plm','x']**2 + df['plm','y']**2 + df['plp','x']**2 + df['plp','y']**2


cuts_dict = {
    'cut1' : r'cos $\theta_{ee, beam}$ > 0.99',
    'cut2' : r'$t_{exp}$ < 0.03',
    'cut3' : r'cos $\theta_{ee}$ > 0',
    'cut4' : r'$p_{ee}$ > 0.15',

}

def compute_selection(df):
    df['cut1', ''] = (df['ee_beam_costheta', ''] > 0.99)
    df['cut2', ''] = (df['experimental_t', ''] < 0.03)
    df['cut3', ''] = (df['ee_costheta', ''] > 0)
    df['cut4', ''] = (df['ee_momentum', ''] > 0.150)
    df['cut_based', ''] = df['cut1', ''] &\
                         df['cut2', ''] &\
                         df['cut3', ''] &\
                         df['cut4', '']


total_pot = 2e21

tpc_length = 100 #cm
fgd_length = 36.5 #cm
p0d_length = 240 #cm

p0d_dimensions = [210.3, 223.9, 240]
detector_splitting = {0: [0, 30.5],
                      1: [30.5, 209.6],
                      2: [209.6, 240.0]}

mol2natoms = 6.02214e23 # Avogadro's number
ton2grams = 1e6
geometry_material = {
    'hydrogen': [0, 1, 2],
    'oxygen': [1],
    'carbon': [0, 1, 2],
    'copper': [1],
    'zinc': [1],
    'lead': [0, 2],
}

material_dict = {
    0.9385: 'hydrogen',
    11.262: 'carbon',
    15.016: 'oxygen',
    59.637921: 'copper',
    61.35913: 'zinc',
    194.4572: 'lead'
}

gev_mass = dict(zip(material_dict.values(), material_dict.keys()))

molar_mass = {
    'hydrogen': 1.00784,
    'carbon': 12.0107,
    'oxygen': 15.999,
    'copper': 63.546,
    'zinc': 65.38,
    'lead': 207.2
}

mass_material = {
    'hydrogen': 3.3*2*molar_mass['hydrogen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen']),
    'oxygen': 3.3*molar_mass['oxygen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen']),
    'carbon': 8.221,
    'copper': 1.315*0.66, # percentage of copper in typical brass
    'zinc': 1.315*0.34, # percentage of zinc in typical brass
    'lead': 3.634,
}

def compute_actual_weights(df):
    ntarget_material = {}
    for material, mass in mass_material.items():
        ntarget_material[material] = mass*ton2grams/molar_mass[material]*mol2natoms

        material_mask = (df['recoil_mass', ''] == gev_mass[material])
        df[material, ''] = material_mask
        df.loc[material_mask, ('total_decay_rate', '')] = df['weight_decay', ''].sum()
        df.loc[material_mask, ('adjusted_weight', '')] = df['weight', ''][material_mask] / df['total_decay_rate', ''][material_mask]
        df.loc[material_mask, ('actual_weight', '')] = df['adjusted_weight', ''][material_mask] * ntarget_material[material] * total_pot

def compute_actual_weights_with_scan(df, which_scan, n_decay_points=30):
    ntarget_material = {}
    for material, mass in mass_material.items():
        ntarget_material[material] = mass*ton2grams/molar_mass[material]*mol2natoms

        material_mask = (df['recoil_mass', ''] == gev_mass[material])
        df[material, ''] = material_mask
        m4_values = df['m4', ''][material_mask].values
        mz_values = df['mzprime', ''][material_mask].values
        weight_values = df['weight', ''][material_mask].values
        weight_decay_values = df['weight_decay', ''][material_mask].values

        if which_scan == 'm4':
            m4_span = np.linspace(m4_values.min(), m4_values.max(), n_decay_points)
            gamma_span = kde_interpolation_m4(m4_span, 
                            m4_values, 
                            weight_decay_values)[0]
            df.loc[material_mask, ('total_decay_rate', '')] = np.interp(m4_values, m4_span, gamma_span)
        elif which_scan == 'm4_mz':
            m4_span = np.linspace(m4_values.min(), m4_values.max(), int(n_decay_points/4))
            mz_span = np.linspace(mz_values.min(), mz_values.max(), int(n_decay_points/4))
            aux_grid = np.stack(np.meshgrid(m4_span, mz_span, indexing='ij'), axis=-1)
            aux_values = np.stack([m4_values, mz_values], axis=-1)
            gamma_kde_weights = kde_Nd_weights(aux_grid, 
                            aux_values,
                            smoothing=[0.03, 0.01])
            gamma_grid = np.sum(gamma_kde_weights*weight_decay_values[:, np.newaxis, np.newaxis], axis=0)
            df.loc[material_mask, ('total_decay_rate', '')] = interpn([m4_span, mz_span], gamma_grid, aux_values)
        df.loc[material_mask, ('adjusted_weight', '')] = weight_values / df['total_decay_rate', ''][material_mask]
        df.loc[material_mask, ('actual_weight', '')] = df['adjusted_weight', ''][material_mask] * ntarget_material[material] * total_pot

def compute_interaction_point(df):
    rg = np.random.default_rng()

    df['int_point', 't'] = 0
    df['int_point', 'x'] = rg.uniform(0, p0d_dimensions[0], len(df))
    df['int_point', 'y'] = rg.uniform(0, p0d_dimensions[1], len(df))

    for material_mass, material in material_dict.items():
        material_mask = (df['recoil_mass', ''] == material_mass)
        region = rg.choice(geometry_material[material], len(df))

        for splitting, boundaries in detector_splitting.items():
            region_mask = (region == splitting)
            total_mask = material_mask & region_mask
            df.loc[total_mask, ('int_point', 'z')] = rg.uniform(*boundaries, total_mask.sum())

def compute_decay_point(df, ctaus):
    if type(ctaus) is not list:
        ctaus = [ctaus]
    for ctau in ctaus:
        decay_particle(df, ctau)
        df[f'decay_in_tpc_{ctau}', ''] = (((p0d_length < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + tpc_length)) |
        (p0d_length + tpc_length + fgd_length < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + tpc_length + fgd_length + tpc_length)) |
        (p0d_length + 2*(tpc_length + fgd_length) < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + 2*(tpc_length + fgd_length) + tpc_length)))) &\
        (detector_splitting[0][0] < df[f'decay_point_{ctau}','x']) & (df[f'decay_point_{ctau}','x'] < detector_splitting[0][1]) &\
        (detector_splitting[1][0] < df[f'decay_point_{ctau}','y']) & (df[f'decay_point_{ctau}','y'] < detector_splitting[1][1])

def decay_in_tpc(df, ctau):
    df[f'decay_point_{ctau}', 't'] = df['int_point', 't'] + ctau*df[f'unitary_decay_length', 't']
    df[f'decay_point_{ctau}', 'x'] = df['int_point', 'x'] + ctau*df[f'unitary_decay_length', 'x']
    df[f'decay_point_{ctau}', 'y'] = df['int_point', 'y'] + ctau*df[f'unitary_decay_length', 'y']
    df[f'decay_point_{ctau}', 'z'] = df['int_point', 'z'] + ctau*df[f'unitary_decay_length', 'z']
    out = (((p0d_length < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + tpc_length)) |
        (p0d_length + tpc_length + fgd_length < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + tpc_length + fgd_length + tpc_length)) |
        (p0d_length + 2*(tpc_length + fgd_length) < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + 2*(tpc_length + fgd_length) + tpc_length)))) &\
        (detector_splitting[0][0] < df[f'decay_point_{ctau}','x']) & (df[f'decay_point_{ctau}','x'] < detector_splitting[0][1]) &\
        (detector_splitting[1][0] < df[f'decay_point_{ctau}','y']) & (df[f'decay_point_{ctau}','y'] < detector_splitting[1][1])
    return out

def initialise_df(df, ctaus=None, is_scan=False):
    compute_analysis_variables(df)
    if is_scan is not False:
        compute_actual_weights_with_scan(df, which_scan=is_scan)
    else:
        compute_actual_weights(df)
    compute_interaction_point(df)
    unitary_decay_length(df)
    compute_selection(df)
    df['no_selection', ''] = np.ones(len(df), dtype=bool)
    if ctaus is not None:
        compute_decay_point(df, ctaus)
        for ctau in ctaus:
            df[f'total_selected_{ctau}', ''] = df[f'decay_in_tpc_{ctau}', ''] & df['cut_based', '']

def kde_interpolation_m4(m4, m4_rnd, weights, smoothing=0.03, kernel='gaus'):
    assert kernel in ['gaus', 'unif']
    m4_span = m4_rnd.max() - m4_rnd.min()
    m4 = m4.T * np.ones([len(m4_rnd), len(m4)])
    m4_rnd = m4_rnd.reshape(len(m4_rnd), 1)
    weights = weights.reshape(len(weights), 1)
    m4_dist = (m4 - m4_rnd)
    f_kde_contribution = weights * m4_span
    # import pdb; pdb.set_trace();
    if kernel == 'gaus':
        f_kde_contribution = f_kde_contribution * norm.pdf(m4_dist, 0, smoothing)
    elif kernel == 'unif':
        f_kde_contribution = f_kde_contribution * uniform.pdf(m4_dist, -smoothing/2, smoothing/2)
    f_kde = f_kde_contribution.sum(axis=0)
    var_f_kde = np.sqrt((f_kde_contribution**2).sum(axis=0))
    return np.nan_to_num(f_kde), np.nan_to_num(var_f_kde)

def kde_1d_weights(x, x_i, smoothing=0.03, kernel='gaus'):
    assert kernel in ['gaus', 'unif']
    x_span = x_i.max() - x_i.min()
    x = x.T * np.ones([len(x_i), len(x)])
    x_i = x_i.reshape(len(x_i), 1)
    x_dist = (x - x_i)
    if kernel == 'gaus':
        kde_weights = x_span * norm.pdf(x_dist, 0, smoothing)
    elif kernel == 'unif':
        kde_weights = x_span * uniform.pdf(x_dist, -smoothing/2, smoothing/2)
    return kde_weights

def kde_Nd_weights(x, x_i, smoothing):
    assert x.shape[-1] == x_i.shape[-1] #number of dimensions
    nd = x_i.shape[-1]
    x_span = np.prod([(x_i[:,i].max() - x_i[:,i].min()) for i in range(nd)]) #assumes points are distributed uniformly in a rectangular surface
    x = np.expand_dims(x, axis=0)
    x_i = np.expand_dims(x_i, axis=list(range(1, len(x.shape))))
    x_dist = (x - x_i)
    smoothing = np.diag(smoothing)**2
    random_weights = multivariate_normal.pdf(x_dist, cov=smoothing)
    kde_weights = x_span * random_weights
    return kde_weights

def mu_sigma2_of_theta(df, m4, mz, ctau, smooth_m4, smooth_mz, selection_step='cut_based_geometric'):
    m4_values = df['m4', ''].values
    mz_values = df['mzprime', ''].values
    df_values = np.stack([m4_values, mz_values], axis=-1)
    this_kde_weights = kde_Nd_weights(np.array([m4, mz]), df_values, smoothing=[smooth_m4, smooth_mz])
    weight_values = this_kde_weights * df['actual_weight', ''].values

    if selection_step == 'no_selection':
        pass
    if 'cut_based' in selection_step:
        weight_values *= df['cut_based', ''].values
    if 'geometric' in selection_step:
        weight_values *= decay_in_tpc(df, ctau)

    return weight_values.sum(), (weight_values**2).sum()

def gaussian1d(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)/np.sqrt(2*np.pi)/sigma

def decay_in_tpc_fast(int_x, int_y, int_z, length_x, length_y, length_z, ctau):
    decay_x = int_x + ctau*length_x
    decay_y = int_y + ctau*length_y
    decay_z = int_z + ctau*length_z
    out = (((p0d_length < decay_z) & (decay_z < (p0d_length + tpc_length)) |
        (p0d_length + tpc_length + fgd_length < decay_z) & (decay_z < (p0d_length + tpc_length + fgd_length + tpc_length)) |
        (p0d_length + 2*(tpc_length + fgd_length) < decay_z) & (decay_z < (p0d_length + 2*(tpc_length + fgd_length) + tpc_length)))) &\
        (detector_splitting[0][0] < decay_x) & (decay_x < detector_splitting[0][1]) &\
        (detector_splitting[1][0] < decay_y) & (decay_y < detector_splitting[1][1])
    return out

def mu_sigma2_of_theta_fast(x_0, x_1, x_i_0, x_i_1, span_2d, smoothing_0, smoothing_1, actual_weights):
    total_weights = actual_weights
    mask = total_weights != 0
    total_weights = total_weights[mask]
    x_i_0 = x_i_0[mask]
    x_i_1 = x_i_1[mask]
    total_weights != 0
    random_weights_0 = gaussian1d(x_i_0-x_0, 0, smoothing_0)
    random_weights_1 = gaussian1d(x_i_1-x_1, 0, smoothing_1)
    this_kde_weights = span_2d * random_weights_0 * random_weights_1
    
    weight_values = this_kde_weights * total_weights

    return weight_values.sum(), (weight_values**2).sum()

def mu_sigma2_of_theta_full(x_0, x_1, x_i_0, x_i_1, span_2d, smoothing_0, smoothing_1, actual_weights, ctau, int_x, int_y, int_z, length_x, length_y, length_z, mu):
    geometric_weights = decay_in_tpc_fast(int_x, int_y, int_z, length_x, length_y, length_z, ctau)
    total_weights = actual_weights * geometric_weights
    total_weights = actual_weights
    mask = total_weights != 0
    total_weights = total_weights[mask]
    x_i_0 = x_i_0[mask]
    x_i_1 = x_i_1[mask]
    total_weights != 0
    random_weights_0 = gaussian1d(x_i_0-x_0, 0, smoothing_0)
    random_weights_1 = gaussian1d(x_i_1-x_1, 0, smoothing_1)
    this_kde_weights = span_2d * random_weights_0 * random_weights_1
    
    weight_values = this_kde_weights * total_weights * mu

    return weight_values.sum(), (weight_values**2).sum()

def mu_sigma2_of_theta_debug(x_0, x_1, x_i_0, x_i_1, span_2d, smoothing_0, smoothing_1, actual_weights, ctau, int_x, int_y, int_z, length_x, length_y, length_z, mu):
    geometric_weights = decay_in_tpc_fast(int_x, int_y, int_z, length_x, length_y, length_z, ctau)
    total_weights = actual_weights * geometric_weights
    total_weights = actual_weights
    mask = total_weights != 0
    total_weights = total_weights[mask]
    x_i_0 = x_i_0[mask]
    x_i_1 = x_i_1[mask]
    total_weights != 0
    random_weights_0 = gaussian1d(x_i_0-x_0, 0, smoothing_0)
    random_weights_1 = gaussian1d(x_i_1-x_1, 0, smoothing_1)
    this_kde_weights = span_2d * random_weights_0 * random_weights_1
    
    weight_values = this_kde_weights * total_weights * mu

    return weight_values

def gamma_light(m4, mz, Valpha4):
    return Valpha4/2 *m4**3/mz**2 * (1-mz**2/m4**2)**2 * (0.5+mz**2/m4**2)

def gamma_zprime_light(m4, mz, alphaepsilon2):
    r = (m4/mz)**2
    return alphaepsilon2/3.0*mz*(1.0+2.0*r)*np.sqrt(1.0-4*r)

def ctau_light(m4, mz, Valpha4):
    aux = Valpha4/2 * m4**3/mz**2 * (1-mz**2/m4**2)**2 * (0.5+mz**2/m4**2)
    return 197.3 * 10**(-16) / aux

def gamma_heavy_contact(m4, mz, Valpha4_alphaepsilon2):
    return Valpha4_alphaepsilon2/(24 * np.pi) * m4**5/mz**4*np.heaviside(mz - m4,0)

# there is a cancellation for small r that holds up to 4th order 
# so I avoid instability by expanding when r is small
def gamma_heavy(m4, mz, Valpha4_alphaepsilon2):
    r=(m4/mz)**2
    gamma = Valpha4_alphaepsilon2/12.0/np.pi/r**2 * m4
    piece = (6*(r -  r**2/2.0 - np.log((1.0/(1.0-r))**(1 - r)) )- r**3)*np.heaviside(r-0.01,0)+r**4/2 * np.heaviside(0.01-r, 1)
    gamma *=  piece
    return gamma*np.heaviside(mz - m4,0)

def gamma_general(m4,mz,Valpha4alphaepsilon2,GammaZprime):
    return -(Valpha4alphaepsilon2*(2*GammaZprime*(m4*m4)*mz*(m4*m4 - 2*(mz*mz)) + (m4*m4*m4*m4*m4*m4 + 3*(m4*m4)*(mz*mz)*(GammaZprime*GammaZprime - mz*mz) + 2*(mz*mz*mz*mz)*(-3*(GammaZprime*GammaZprime) + mz*mz))*np.arctan(GammaZprime/mz) + (m4*m4*m4*m4*m4*m4 + 3*(GammaZprime*GammaZprime)*(m4*m4)*(mz*mz) + 2*(mz*mz*mz*mz*mz*mz))*np.arctan((GammaZprime*mz)/(m4*m4 - mz*mz)) + mz*mz*mz*(3*(2*(GammaZprime*GammaZprime) + m4*m4)*mz*np.arctan((GammaZprime*mz)/(-(m4*m4) + mz*mz)) - GammaZprime*(2*(GammaZprime*GammaZprime + 3*(m4*m4) - 3*(mz*mz))*np.log(mz) + (GammaZprime*GammaZprime + 3*(m4*m4))*(np.log(GammaZprime*GammaZprime + mz*mz) - np.log(GammaZprime*GammaZprime*(mz*mz) + (m4*m4 - mz*mz)*(m4*m4 - mz*mz))) + 3*(mz*mz)*np.log(mz*mz + (m4*m4*m4*m4 - 2*(m4*m4)*(mz*mz))/(GammaZprime*GammaZprime + mz*mz))))))/(12.*GammaZprime*(m4*m4*m4)*mz*np.pi)

# This still assumes contact interaction
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

def ctau_heavy(m4, mz, Valpha4_alphaepsilon2):
    aux =  gamma_heavy(m4, mz, Valpha4_alphaepsilon2)
    return 197.3 * 10**(-16) / aux

def gamma_to_ctau(gamma):
    '''Convert gamma [GeV] to ctau [cm]'''
    return 197.3 * 10**(-16) / gamma

def points_on_triangle(N_points, m4_limits, mz_limits, hierarchy='heavy'):
    rvs = np.random.random((N_points, 2))
    if hierarchy == 'heavy':
        rvs = np.where(rvs[:, 0, None]<rvs[:, 1, None], rvs, rvs[:, ::-1])
    elif hierarchy == 'light':
        rvs = np.where(rvs[:, 0, None]>rvs[:, 1, None], rvs, rvs[:, ::-1])
    
    return np.array((m4_limits[0], mz_limits[0])) + rvs*(m4_limits[1]-m4_limits[0], mz_limits[1]-mz_limits[0])