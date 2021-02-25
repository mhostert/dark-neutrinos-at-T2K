import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import interpn
import pandas as pd
import matplotlib.pyplot as plt
from fourvec import *
from parameters_dict import *

def gamma_light(m4, mz, Valpha4):
    return Valpha4/2 *m4**3/mz**2 * (1-mz**2/m4**2)**2 * (0.5+mz**2/m4**2)

def ctau_light(m4, mz, Valpha4):
    aux = Valpha4/2 * m4**3/mz**2 * (1-mz**2/m4**2)**2 * (0.5+mz**2/m4**2)
    return 197.3 * 10**(-16) / aux

def gamma_heavy(m4, mz, Valpha4_alphaepsilon2):
    return Valpha4_alphaepsilon2/(24 * np.pi) * m4**5/mz**4

def ctau_heavy(m4, mz, Valpha4_alphaepsilon2):
    aux =  Valpha4_alphaepsilon2/(24 * np.pi) * m4**5/mz**4
    return 197.3 * 10**(-16) / aux

def gamma_to_ctau(gamma):
    '''Convert gamma [GeV] to ctau [cm]'''
    return 197.3 * 10**(-16) / gamma

def gaussian1d(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)/np.sqrt(2*np.pi)/sigma

def points_on_triangle(N_points, m4_limits, mz_limits, hierarchy='heavy'):
    rvs = np.random.random((N_points, 2))
    if hierarchy == 'heavy':
        rvs = np.where(rvs[:, 0, None]<rvs[:, 1, None], rvs, rvs[:, ::-1])
    elif hierarchy == 'light':
        rvs = np.where(rvs[:, 0, None]>rvs[:, 1, None], rvs, rvs[:, ::-1])
    
    return np.array((m4_limits[0], mz_limits[0])) + rvs*(m4_limits[1]-m4_limits[0], mz_limits[1]-mz_limits[0])


class exp_analysis(object):
    
    def __init__(self, hierarchy, base_folder='../data/nd280_nu/3plus1/'):
        self.__dict__.update(physics_parameters[hierarchy])
        self.hierarchy = hierarchy
        self.base_folder = base_folder
        self.dfs = {}
        
    def load_df_base(self, n_evt=1000000):
        self.n_evt = n_evt
        self.df_base = pd.read_pickle(f'{self.base_folder}scan/{self.hierarchy}_mediator/{self.m4_limits[0]}_m4_{self.m4_limits[1]}_{self.mz_limits[0]}_mzprime_{self.mz_limits[1]}_nevt_{self.n_evt}.pckl')
        self.initialise_df(self.df_base, which_scan='m4_mz')
        
    def load_df(self, m4, mz):
        self.dfs[(m4, mz)] = pd.read_pickle(f'{self.base_folder}m4_{m4}_mzprime_{mz}/MC_m4_{m4}_mzprime_{mz}.pckl')
        exp_analysis.initialise_df(self.dfs[(m4, mz)], None)
        return self.dfs[(m4, mz)]
        
    @staticmethod
    def initialise_df(df, which_scan):
        exp_analysis.compute_analysis_variables(df)
        exp_analysis.compute_actual_weights(df, which_scan)
        exp_analysis.compute_interaction_point(df)
        exp_analysis.unitary_decay_length(df)
        exp_analysis.compute_selection(df)
    
    @staticmethod
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
    @staticmethod
    def compute_actual_weights(df, which_scan=None, smooth_pars_decaywidth=[0.03, 0.01], n_points_decaywidth_interpolation=30):
        ntarget_material = {}
        for material, mass in mass_material.items():
            ntarget_material[material] = mass*ton2grams/molar_mass[material]*mol2natoms

            material_mask = (df['recoil_mass', ''] == gev_mass[material])
            df[material, ''] = material_mask
            m4_values = df['m4', ''][material_mask].values
            mz_values = df['mzprime', ''][material_mask].values
            weight_values = df['weight', ''][material_mask].values
            weight_decay_values = df['weight_decay', ''][material_mask].values
            
            if which_scan == None:
                df.loc[material_mask, ('total_decay_rate', '')] = weight_decay_values.sum()                
            if which_scan == 'm4':
                m4_span = np.linspace(m4_values.min(), m4_values.max(), n_points_decaywidth_interpolation)
                gamma_span = kde_interpolation_m4(m4_span, 
                                m4_values, 
                                weight_decay_values)[0]
                df.loc[material_mask, ('total_decay_rate', '')] = np.interp(m4_values, m4_span, gamma_span)
            elif which_scan == 'm4_mz':
                weight_values = weight_values / mz_values**8
                weight_decay_values = weight_decay_values / mz_values**4
                m4_span = np.linspace(m4_values.min(), m4_values.max(), int(n_points_decaywidth_interpolation/4))
                mz_span = np.linspace(mz_values.min(), mz_values.max(), int(n_points_decaywidth_interpolation/4))
                aux_grid = np.stack(np.meshgrid(m4_span, mz_span, indexing='ij'), axis=-1)
                aux_values = np.stack([m4_values, mz_values], axis=-1)
                gamma_kde_weights = exp_analysis.kde_Nd_weights(aux_grid, 
                                aux_values,
                                smoothing=smooth_pars_decaywidth)
                gamma_grid = np.sum(gamma_kde_weights*weight_decay_values[:, np.newaxis, np.newaxis], axis=0)
                df.loc[material_mask, ('total_decay_rate', '')] = interpn([m4_span, mz_span], gamma_grid, aux_values)
            df.loc[material_mask, ('adjusted_weight', '')] = weight_values / df['total_decay_rate', ''][material_mask]
            df.loc[material_mask, ('actual_weight', '')] = df['adjusted_weight', ''][material_mask] * ntarget_material[material] * total_pot
        
    @staticmethod
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

    @staticmethod
    def unitary_decay_length(df):
        p3dark = np.sqrt(dot3_df(df['pdark'], df['pdark']))
        mdark = inv_mass(df['pdark'])
        betagamma = p3dark/mdark
        gamma = df['pdark', 't']/mdark
        
        d_decay = np.random.exponential(scale=betagamma) # it's for ctau=1
        df[f'unitary_decay_length', 'x'] = d_decay*df['pdark', 'x']/p3dark
        df[f'unitary_decay_length', 'y'] = d_decay*df['pdark', 'y']/p3dark
        df[f'unitary_decay_length', 'z'] = d_decay*df['pdark', 'z']/p3dark

    @staticmethod
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
        
    @staticmethod
    def decay_in_tpc(df, ctaus):
        if type(ctaus) is not list:
            ctaus = [ctaus]
        for ctau in ctaus:
            decay_particle(df, ctau)
            df[f'decay_in_tpc_{ctau}', ''] = (((p0d_length < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + tpc_length)) |
            (p0d_length + tpc_length + fgd_length < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + tpc_length + fgd_length + tpc_length)) |
            (p0d_length + 2*(tpc_length + fgd_length) < df[f'decay_point_{ctau}','z']) & (df[f'decay_point_{ctau}','z'] < (p0d_length + 2*(tpc_length + fgd_length) + tpc_length)))) &\
            (detector_splitting[0][0] < df[f'decay_point_{ctau}','x']) & (df[f'decay_point_{ctau}','x'] < detector_splitting[0][1]) &\
            (detector_splitting[1][0] < df[f'decay_point_{ctau}','y']) & (df[f'decay_point_{ctau}','y'] < detector_splitting[1][1])

    @staticmethod
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

    @staticmethod
    def compute_selection(df):
        df['cut1', ''] = (df['ee_beam_costheta', ''] > 0.99)
        df['cut2', ''] = (df['experimental_t', ''] < 0.03)
        df['cut3', ''] = (df['ee_costheta', ''] > 0)
        df['cut4', ''] = (df['ee_momentum', ''] > 0.150)
        df['cut_based', ''] = df['cut1', ''] &\
                            df['cut2', ''] &\
                            df['cut3', ''] &\
                            df['cut4', '']


    @staticmethod
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

    @staticmethod
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
        else:
            return None
        return kde_weights

    @staticmethod
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

    @staticmethod
    def mu_sigma2_of_theta(df, m4, mz, ctau, smooth_m4, smooth_mz, selection_step='cut_based_geometric'):
        m4_values = df['m4', ''].values
        mz_values = df['mzprime', ''].values
        df_values = np.stack([m4_values, mz_values], axis=-1)
        this_kde_weights = exp_analysis.kde_Nd_weights(np.array([m4, mz]), df_values, smoothing=[smooth_m4, smooth_mz])
        weight_values = this_kde_weights * df['actual_weight', ''].values

        if selection_step == 'no_selection':
            pass
        if 'cut_based' in selection_step:
            weight_values *= df['cut_based', ''].values
        if 'geometric' in selection_step:
            weight_values *= exp_analysis.decay_in_tpc(df, ctau)

        return weight_values.sum(), (weight_values**2).sum()

    @staticmethod
    def mu_sigma2_of_theta_no_geometry(x_0, x_1, x_i_0, x_i_1, span_2d, smoothing_0, smoothing_1, actual_weights):
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

    @staticmethod
    def mu_sigma2_of_theta_full(x_0, x_1, x_i_0, x_i_1, span_2d, smoothing_0, smoothing_1, actual_weights, ctau, int_x, int_y, int_z, length_x, length_y, length_z, mu):
        geometric_weights = exp_analysis.decay_in_tpc_fast(int_x, int_y, int_z, length_x, length_y, length_z, ctau)
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
    
    def compare_distributions(self, m4, mz, variable):
        assert (m4, mz) in self.dfs.keys()

        