import itertools
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from scipy.interpolate import interpn
from scipy.integrate import dblquad
import pandas as pd
import matplotlib.pyplot as plt
from fourvec import *
from parameters_dict import *

def gamma_light(m4, mz, Valpha4):
    return Valpha4/2 *m4**3/mz**2 * (1-mz**2/m4**2)**2 * (0.5+mz**2/m4**2)

def ctau_light(m4, mz, Valpha4):
    aux = Valpha4/2 * m4**3/mz**2 * (1-mz**2/m4**2)**2 * (0.5+mz**2/m4**2)
    return 197.3 * 10**(-16) / aux

def gamma_heavy_contact(m4, mz, Valpha4_alphaepsilon2):
    return Valpha4_alphaepsilon2/(24 * np.pi) * m4**5/mz**4*np.heaviside(mz - m4,0)

# there is a cancellation for small r that holds up to 4th order 
# so I avoid instability by expanding when r is small
# it needs to be adjusted to handle well m4 = mz
def gamma_heavy(m4, mz, Valpha4_alphaepsilon2):
    r = ((m4/mz)**2)
    gamma = Valpha4_alphaepsilon2/12.0/np.pi/r**2 * m4
    piece = (6*(r -  r**2/2.0 - np.log((1.0/(1.0-r))**(1 - r)) )- r**3)*np.heaviside(r-0.01,0)\
    +r**4/2 * np.heaviside(0.01-r, 1)
    gamma *=  piece
    return gamma

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
    aux =  Valpha4_alphaepsilon2/(24 * np.pi) * m4**5/mz**4
    return 197.3 * 10**(-16) / aux

def gamma_to_ctau(gamma):
    '''Convert gamma [GeV] to ctau [cm]'''
    return 197.3 * 10**(-16) / gamma

def gaussian1d(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)/np.sqrt(2*np.pi)/sigma

def epa_kernel2d(x, sigma):
    return np.where((x[..., 0]/sigma[0])**2 + (x[..., 1]/sigma[1])**2 > 1,
                    0,
                    (1 - (x[..., 0]/sigma[0])**2 - (x[..., 1]/sigma[1])**2)/(4*sigma[0]*sigma[1]*(1-1/np.sqrt(3))))


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
        
    def load_df_base(self, n_evt=1000000, filename=None):
        self.n_evt = n_evt
        if filename is None:
            self.df_base = pd.read_pickle(f'{self.base_folder}scan/{self.hierarchy}_mediator/{self.m4_limits[0]}_m4_{self.m4_limits[1]}_{self.mz_limits[0]}_mzprime_{self.mz_limits[1]}_nevt_{self.n_evt}.pckl')
        else:
            self.df_base = pd.read_pickle(filename)
        self.initialise_df(self.df_base, which_scan='m4_mz')
    
    def load_df(self, m4, mz):
        self.dfs[(m4, mz)] = pd.read_pickle(f'{self.base_folder}m4_{m4}_mzprime_{mz}/MC_m4_{m4}_mzprime_{mz}.pckl')
        self.initialise_df(self.dfs[(m4, mz)], None)
        return self.dfs[(m4, mz)]
    
    def load_grid_dfs(self):
        for m4, mz in itertools.product(self.m4_scan, self.mz_scan):
            if ((self.hierarchy == 'heavy') and (m4 >= mz)) or ((self.hierarchy == 'light') and (m4 <= mz)):
                continue
            else:
                self.load_df(m4, mz)
    
    def initialise_df(self, df, which_scan):
        self.compute_analysis_variables(df)
        self.compute_actual_weights(df, which_scan)
        self.compute_interaction_point(df)
        self.unitary_decay_length(df)
        self.compute_selection(df)
        
        self.m4_values = self.df_base['m4', ''].values
        self.mz_values = self.df_base['mzprime', ''].values
        self.m4mz_values = np.stack([self.m4_values, self.mz_values], axis=-1)
        self.actual_weight_values = self.df_base['actual_weight', ''].values
        
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

    def compute_actual_weights(self, df, which_scan=None, with_decay_formula=True, smooth_pars_decaywidth=[0.005, 0.05], kernel='epa', n_points_decaywidth_interpolation=30):
        m4_values = df['m4', ''].values
        mz_values = df['mzprime', ''].values
        weight_values = df['weight', ''].values
        
        #first fix decay rate
        if with_decay_formula:
            if self.hierarchy == 'heavy':
                df['total_decay_rate', ''] = gamma_heavy(m4_values,
                                                         mz_values,
                                                         self.Vmu4_alpha_epsilon2)
            else:
                print('Light hierarchy with formula not supported yet')
                return
        else:
            weight_decay_values = df['weight_decay', ''].values
            if which_scan == None:
                df['total_decay_rate', ''] = weight_decay_values.sum()                
            elif which_scan == 'm4_mz':
                m4_span = np.linspace(m4_values.min(), m4_values.max(), int(n_points_decaywidth_interpolation/4))
                mz_span = np.linspace(mz_values.min(), mz_values.max(), int(n_points_decaywidth_interpolation/4))
                aux_grid = np.stack(np.meshgrid(m4_span, mz_span, indexing='ij'), axis=-1)
                aux_values = np.stack([m4_values, mz_values], axis=-1)
                gamma_kde_weights = exp_analysis.kde_Nd_weights(aux_grid, 
                                aux_values,
                                smoothing=smooth_pars_decaywidth,
                                kernel=kernel)
                gamma_grid = np.sum(gamma_kde_weights*weight_decay_values[:, np.newaxis, np.newaxis], axis=0)
                df['total_decay_rate', ''] = interpn([m4_span, mz_span], gamma_grid, aux_values)
        
        df['adjusted_weight', ''] = weight_values / df['total_decay_rate', '']
        
        ntarget_material = {}
        for material, mass in mass_material.items():
            ntarget_material[material] = mass*ton2grams/molar_mass[material]*mol2natoms
            material_mask = (df['recoil_mass', ''] == gev_mass[material])
            if material_mask.sum() == 0:
                continue
            df[material, ''] = material_mask
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
            exp_analysis.decay_particle(df, ctau)
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
    def kde_Nd_weights(x, x_i, smoothing, kernel='epa'):
        """x is where to evaluate (could be a grid), x_i are the montecarlo points for training"""
        assert x.shape[-1] == x_i.shape[-1] #number of dimensions
        x_i = np.expand_dims(x_i, axis=list(range(1, len(x.shape))))
        x = np.expand_dims(x, axis=0) #add the axis for the number of points over which to sum.
        x_dist = (x - x_i)
        if kernel == 'gaus':
            smoothing = np.diag(smoothing)**2
            kde_weights = multivariate_normal.pdf(x_dist, cov=smoothing)
        elif kernel == 'epa':
            kde_weights = epa_kernel2d(x_dist, smoothing)
        return kde_weights
    
    def kde_on_a_point(self, this_m4mz, smoothing=[0.005, 0.05], kernel='epa'):
        if type(this_m4mz) != np.ndarray:
            this_m4mz = np.array(this_m4mz)
        return self.actual_weight_values * self.kde_Nd_weights(this_m4mz, 
                                                               self.m4mz_values, 
                                                               smoothing, 
                                                               kernel)

    def kde_on_a_grid(self, m4_span, mz_span, smoothing=[0.005, 0.05], kernel='epa'):
        grid_to_eval = np.stack(np.meshgrid(m4_span, mz_span, indexing='ij'), axis=-1)
        this_kde_weights = self.kde_Nd_weights(grid_to_eval, 
                                               self.m4mz_values, 
                                               smoothing, 
                                               kernel)

        return self.actual_weight_values[:, np.newaxis, np.newaxis] * this_kde_weights
    
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
    def mu_sigma2_of_theta_no_geometry(x_0, x_1, x_i_0, x_i_1, smoothing_0, smoothing_1, actual_weights):
        total_weights = actual_weights
        mask = total_weights != 0
        total_weights = total_weights[mask]
        x_i_0 = x_i_0[mask]
        x_i_1 = x_i_1[mask]
        total_weights != 0
        random_weights_0 = gaussian1d(x_i_0-x_0, 0, smoothing_0)
        random_weights_1 = gaussian1d(x_i_1-x_1, 0, smoothing_1)
        this_kde_weights = random_weights_0 * random_weights_1
        
        weight_values = this_kde_weights * total_weights

        return weight_values.sum(), (weight_values**2).sum()

    @staticmethod
    def mu_sigma2_of_theta_full(x_0, x_1, x_i_0, x_i_1, smoothing_0, smoothing_1, actual_weights, ctau, int_x, int_y, int_z, length_x, length_y, length_z, mu):
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
        this_kde_weights = random_weights_0 * random_weights_1
        
        weight_values = this_kde_weights * total_weights * mu

        return weight_values.sum(), (weight_values**2).sum()