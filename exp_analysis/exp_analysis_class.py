from copy import deepcopy
import itertools
import numpy as np
from scipy.interpolate import interpn
import pandas as pd

from sklearn.neighbors import BallTree
from kde_utils import kde_Nd_weights, log_distance

from Likelihood import LEff_v

from const import alphaQED
from fourvec import *
from parameters_dict import *
from const import *
from ctau_utils import *

droplist = ['plm_t', 'plm_x', 'plm_y', 'plm_z', 'plp_t', 'plp_x', 'plp_y', 'plp_z',
            'pnu_t', 'pnu_x', 'pnu_y', 'pnu_z', 'pHad_t', 'pHad_x', 'pHad_y',
            'pHad_z', 'weight_decay', 'regime', 'pee_t',
            'pdark_t', 'pee_x', 'pdark_x', 'pee_y', 'pdark_y', 'pee_z', 'pdark_z',
            'recoil_mass']

def compute_likelihood_from_retrieved(out_events_weights, exp_analysis_obj, analysis, like_normalized=True):
    if analysis['selection'] is not None:
        selection_mask = out_events_weights['df'].eval(analysis['selection'])
        aux_df = out_events_weights['df'][selection_mask]
        all_weights = out_events_weights['weights'][selection_mask]
    else:
        aux_df = out_events_weights['df']
        all_weights = out_events_weights['weights']
    pot_ntarget_weights = exp_analysis_obj.compute_pot_ntarget_weights(aux_df, 
                                                                       analysis['n_target'], 
                                                                       analysis['pot'])
    all_weights = all_weights.T * pot_ntarget_weights * analysis['efficiency']
    return exp_analysis_obj.compute_likelihood(aux_df, all_weights.T, analysis, like_normalized)

def full_likelihood(m4, mz, alpha_dark, epsilon, Umu4_2, Ud4_2, exp_analyses_objects, hierarchy, D_or_M, analyses, like_normalized=False):
    leff, mu, sigma2 = 0, 0, 0
    for analysis in analyses:
        for flux, sub_analysis in analysis.items():
            print(flux)
            this_analysis_object = exp_analyses_objects[f'{hierarchy}_{D_or_M}_{flux}']
            out = this_analysis_object.compute_likelihood_from_pars(this_analysis_object.df_base, 
                                                            m4, mz, alpha_dark, epsilon, Umu4_2, Ud4_2, sub_analysis,
                                                                   like_normalized=like_normalized)
            leff += out[0]
            mu += out[1]
            sigma2 += out[2]
    if like_normalized:
        leff -= leff.min()
    return leff, mu, sigma2

class exp_analysis(object):
    
    def __init__(self, hierarchy, D_or_M, flux='FHC', base_folder='../data'):
        self.__dict__.update(physics_parameters[hierarchy])
        self.hierarchy = hierarchy
        self.D_or_M = D_or_M
        self.flux = flux
        if  flux == 'FHC':
            self.base_folder = base_folder + '/nd280_nu'
        elif flux == 'RHC':
            self.base_folder = base_folder + '/nd280_nubar'
        self.base_folder += '/3plus1/'
        self.dfs = {}

    def load_df_base(self, n_evt=1000000, filename=None, build_ball_tree=False, distance=default_kde_pars['distance'], smearing=False, smearing_folder='smearing_matrices/'):
        self.n_evt = n_evt
        print("loading df base")
        if filename is None:
            self.df_base = pd.read_pickle(f'{self.base_folder}scan/{self.hierarchy}_{self.D_or_M}/{self.m4_limits[0]}_m4_{self.m4_limits[1]}_{self.mz_limits[0]}_mzprime_{self.mz_limits[1]}_nevt_{self.n_evt}.pckl')
        else:
            self.df_base = pd.read_pickle(filename)
        print("initialising df base")
        self.initialise_df(self.df_base, which_scan='m4_mz')

        if build_ball_tree:
            if distance == 'lin':
                self.ball_tree = BallTree(self.m4mz_values)
            elif distance == 'log':
                self.ball_tree = BallTree(self.m4mz_values, metric='pyfunc', func=log_distance)
        
        if smearing:
            self.smearing_p = np.load(smearing_folder + 'Momentum_matrix.npy')
            self.p_binning_true = np.load(smearing_folder + 'Momentum_binning_true.npy')/1000
            self.p_binning_reco = np.load(smearing_folder + 'Momentum_binning_reco.npy')/1000
            self.smearing_theta = np.load(smearing_folder + 'Angle_matrix.npy')
            self.theta_binning_true = np.load(smearing_folder + 'Angle_binning_true.npy')
            self.theta_binning_reco = np.load(smearing_folder + 'Angle_binning_reco.npy')
            self.setResolution(self.smearing_p, self.p_binning_true, self.p_binning_reco, 
                                          self.smearing_theta, self.theta_binning_true, self.theta_binning_reco)
            
    def load_df(self, m4, mz):
        print(f"loading df {m4}, {mz}")
        self.dfs[(m4, mz)] = pd.read_pickle(f'{self.base_folder}m4_{m4}_mzprime_{mz}_{self.hierarchy}_{self.D_or_M}/MC_m4_{m4}_mzprime_{mz}.pckl')
        print(f"initialising df {m4}, {mz}")
        self.initialise_df(self.dfs[(m4, mz)])
    
    def load_grid_dfs(self):
        for m4, mz in itertools.product(self.m4_scan, self.mz_scan):
            if ((self.hierarchy == 'heavy') and (m4 >= mz)) or ((self.hierarchy == 'light') and (m4 <= mz)):
                continue
            else:
                self.load_df(m4, mz)

    def initialise_df(self, df, which_scan=None):
        self.compute_analysis_variables(df)
        self.adjust_weights_decay_width(df, which_scan)
        self.create_material_mask(df)
        self.compute_selection(df)
        
        if self.hierarchy == 'heavy':
            self.compute_interaction_point(df)
            self.unitary_decay_length(df)
            self.compute_decay_integral(df)
        
        # flatten index of pandas multiindex
        df.columns = ['_'.join(col) if (col[1]!='') else col[0] for col in df.columns.values]
        df.drop(droplist, axis=1, inplace=True)
        
        for column in df.columns:
            if (df[column].dtype == 'float') and ('weight' not in column):
                df[column] = pd.to_numeric(df[column], downcast='float')
                
        if which_scan == 'm4_mz':
            self.m4_values = self.df_base['m4'].values
            self.mz_values = self.df_base['mzprime'].values
            self.m4mz_values = np.stack([self.m4_values, self.mz_values], axis=-1)
            # self.actual_weight_values = self.df_base['actual_weight'].values

    def adjust_weights_decay_width(self, df, which_scan=None):
        m4_values = df['m4', ''].values
        mz_values = df['mzprime', ''].values
        weight_values = df['weight', ''].values
        
        if which_scan == None:
            df['total_decay_rate', ''] = gamma_general(m4_values[0],
                                                        mz_values[0],
                                                        self.Vmu4_alpha_epsilon2,
                                                        D_or_M=self.D_or_M)
        elif which_scan == 'm4_mz':
            df['total_decay_rate', ''] = gamma_general(m4_values,
                                                        mz_values,
                                                        self.Vmu4_alpha_epsilon2,
                                                        D_or_M=self.D_or_M)

        df['generator_weight', ''] = weight_values / df['total_decay_rate', '']
        
    def setResolution(self, smearing_p, p_binning_true, p_binning_reco, smearing_theta, theta_binning_true, theta_binning_reco):    
        self.smearing_theta = self.prepareSmearingMatrix(smearing_theta, theta_binning_true, theta_binning_reco)
        self.df_base['em_beam_theta_reco'] = self.true2reco(self.df_base['em_beam_theta'], 
                                                     self.smearing_theta, theta_binning_true, theta_binning_reco)
        self.df_base['ep_beam_theta_reco'] = self.true2reco(self.df_base['ep_beam_theta'], 
                                                     self.smearing_theta, theta_binning_true, theta_binning_reco)
        self.smearing_p = self.prepareSmearingMatrix(smearing_p, p_binning_true, p_binning_reco)
        self.df_base['em_energy_reco'] = self.true2reco(self.df_base['em_energy'], 
                                                     self.smearing_p, p_binning_true, p_binning_reco)
        self.df_base['ep_energy_reco'] = self.true2reco(self.df_base['ep_energy'], 
                                                     self.smearing_p, p_binning_true, p_binning_reco)
        
        self.df_base['emp_opening_cos_angle_reco'] = cos_opening_angle(self.df_base['em_beam_theta_reco'],
                                                                       self.df_base['em_beam_phi'], 
                                                                       self.df_base['ep_beam_theta_reco'], 
                                                                       self.df_base['ep_beam_phi'])
        
        self.df_base['ee_mass_reco'] = inv_mass_massless(self.df_base['em_energy_reco'],
                                                         self.df_base['ep_energy_reco'],
                                                         self.df_base['emp_opening_cos_angle_reco'])
    
    @staticmethod
    def prepareSmearingMatrix(smearing_matrix, binning_true, binning_reco):
        # normalise smearing_matrices so that each column sums to 1
        smearing_matrix_sum = smearing_matrix.sum(axis=1)
        bin_centers = (binning_true[1:] + binning_true[:-1])/2
        indices_x = np.where(smearing_matrix_sum == 0)[0]
        indices_y = np.digitize(bin_centers[indices_x], binning_reco) - 1 #-1 becuase no entry should be in the underflow
        smearing_matrix[indices_x, indices_y] = 1
        return (smearing_matrix.T/smearing_matrix.sum(axis=1)).T
    
    @staticmethod
    def true2reco(true_entries, smearing_matrix, binning_true, binning_reco):
        # check that the smearing matrix columns sums to 1 - i.e. it's a good p(y|x)
        np.testing.assert_allclose(smearing_matrix.sum(axis=1), np.ones(smearing_matrix.shape[0]), rtol=1e-5)
        
        # create cumulative F(y|x)
        smearing_matrix_cumsum = smearing_matrix.cumsum(axis=1).astype('float16') #astype float16 to avoid roundoff errors
        
        # digitize true entries - it should give indices between 0 and 
        true_entries_digitized = np.digitize(true_entries, binning_true) - 1 #-1 becuase no entry should be in the underflow
        
        overflow_indices = np.where(true_entries_digitized == (len(binning_true) - 1))[0]
        true_entries_digitized[overflow_indices] = len(binning_true) - 2
        
        aux_p = np.random.rand(len(true_entries))
        reco_entries_digitized = (smearing_matrix_cumsum[true_entries_digitized] < aux_p[:,None]).sum(axis=1)
        
        bin_centers = (binning_reco[1:] + binning_reco[:-1])/2
        out = bin_centers[reco_entries_digitized]
        out[overflow_indices] = true_entries[overflow_indices]
        return out
    
    @staticmethod
    def compute_analysis_variables(df):
        for comp in ['t','x','y','z']:
            df['pee', comp] = df['plm', comp] + df['plp', comp]
            df['pdark', comp] = df['plm', comp] + df['plp', comp] + df['pnu', comp]

        df['recoil_mass', ''] = inv_mass(df['pHad']).round(6)
        # e+e- cone vars
        df['ee_mass', ''] = inv_mass(df['pee'])
        df['ee_energy', ''] = df['pee', 't']
        df['ee_momentum', ''] = np.sqrt(dot3_df(df['pee'], df['pee']))    
        df['ee_energy_asymetry', ''] = (df['plm', 't']-df['plp', 't'])/(df['plp', 't']+df['plm', 't'])
        df['ee_costheta', ''] = costheta(df['plm'], df['plp'])
        df['ee_theta', ''] = np.arccos(costheta(df['plm'], df['plp']))
        df['ee_beam_costheta', ''] = df['pee', 'z']/np.sqrt(dot3_df(df['pee'], df['pee']))
        df['ee_beam_theta', ''] = np.arccos(df['pee', 'z']/np.sqrt(dot3_df(df['pee'], df['pee'])))
        # dark nu vars
        df['nu_dark_beam_costheta', ''] = df['pdark', 'z']/np.sqrt(dot3_df(df['pdark'], df['pdark']))
        # e- vars
        df['em_energy', ''] = df['plm', 't']
        df['em_beam_costheta', ''] = df['plm', 'z']/np.sqrt(dot3_df(df['plm'], df['plm']))
        df['em_beam_theta', ''] = np.arccos(df['em_beam_costheta', ''])
        df['em_beam_phi', ''] = np.arctan2(df['plm', 'y'], df['plm', 'x'])
        
        # e+ vars
        df['ep_energy', ''] = df['plp', 't']
        df['ep_beam_costheta', ''] = df['plp', 'z']/np.sqrt(dot3_df(df['plp'], df['plp']))
        df['ep_beam_theta', ''] = np.arccos(df['ep_beam_costheta', ''])
        df['ep_beam_phi', ''] = np.arctan2(df['plp', 'y'], df['plp', 'x'])
        
        ## e+e- vars
        df['emp_opening_cos_angle'] = cos_opening_angle(df['em_beam_theta'],
                                                        df['em_beam_phi'], 
                                                        df['ep_beam_theta'],
                                                        df['ep_beam_phi'])
        df['ee_mass_from_angle_true'] = inv_mass_massless(df['em_energy'],
                                                         df['ep_energy'],
                                                         df['emp_opening_cos_angle'])
        
        # high level vars
        df['experimental_t', ''] = (df['plm','t'] - df['plm','z'] + df['plp','t'] - df['plp','z'])**2 +\
                                    df['plm','x']**2 + df['plm','y']**2 + df['plp','x']**2 + df['plp','y']**2
        
        df['p3dark', ''] = np.sqrt(dot3_df(df['pdark'], df['pdark']))
        df['mdark', ''] = inv_mass(df['pdark'])
        df['betagamma', ''] = df['p3dark', '']/df['mdark', '']
    
    @staticmethod
    def create_material_mask(df):
        for material, atomic_mass in atomic_mass_gev.items():
            material_mask = (df['recoil_mass', ''] == atomic_mass)
            if material_mask.sum() == 0:
                continue
            df[material, ''] = material_mask
    
    @staticmethod
    def compute_pot_ntarget_weights(df, ntarget_per_material, pot):
        out_weights = np.zeros(len(df))
        for material, ntarget in ntarget_per_material.items():
            out_weights[df[material]] = ntarget * pot
        return out_weights

    @staticmethod
    def compute_interaction_point(df):
        rg = np.random.default_rng()
        
        for material in atomic_mass_gev.keys():
            material_mask = df[material, '']
            region = rg.choice(geometry_material[material], len(df), p=mass_weights[material])

            for splitting in detector_splitting_x.keys():
                region_mask = (region == splitting)
                total_mask = material_mask & region_mask
                df.loc[total_mask, ('int_point', 'x')] = rg.uniform(*(detector_splitting_x[splitting]), total_mask.sum())
                df.loc[total_mask, ('int_point', 'y')] = rg.uniform(*(detector_splitting_y[splitting]), total_mask.sum())
                df.loc[total_mask, ('int_point', 'z')] = rg.uniform(*(detector_splitting_z[splitting]), total_mask.sum())
                
    @staticmethod
    def is_point_in_tpc(x, y, z):
        is_in_x = (tpc_fiducial_volume_endpoints[0][0] < x) & (x < tpc_fiducial_volume_endpoints[0][1])
        is_in_y = (tpc_fiducial_volume_endpoints[1][0] < y) & (y < tpc_fiducial_volume_endpoints[1][1])
        is_in_z_tpc1 = (tpc_fiducial_volume_endpoints[2][0] < z) & (z < tpc_fiducial_volume_endpoints[2][1])
        z_tpc2 = z - tpc_outer_volume[2] - fgd_outer_volume[2]
        is_in_z_tpc2 = (tpc_fiducial_volume_endpoints[2][0] < z_tpc2) & (z_tpc2 < tpc_fiducial_volume_endpoints[2][1])
        z_tpc3 = z - 2 * (tpc_outer_volume[2] - fgd_outer_volume[2])
        is_in_z_tpc3 = (tpc_fiducial_volume_endpoints[2][0] < z_tpc3) & (z_tpc3 < tpc_fiducial_volume_endpoints[2][1])
        return is_in_x & is_in_y & (is_in_z_tpc1 | is_in_z_tpc2 | is_in_z_tpc3)

    @staticmethod
    def is_point_in_tpc_general(c_s, coords, tpc_index=0):
        assert len(c_s) == len(coords)
        out = np.ones(c_s[0].shape, dtype=bool)
        for c, coord in zip(c_s, coords):
            if coord == 2:
                out &= (((tpc_fiducial_volume_endpoints[coord][0] + tpc_index * (tpc_outer_volume[2] + fgd_outer_volume[2])) < c) &\
                       (c < (tpc_fiducial_volume_endpoints[coord][1] + tpc_index * (tpc_outer_volume[2] + fgd_outer_volume[2]))))
            else:
                out &= ((tpc_fiducial_volume_endpoints[coord][0] < c) &\
                          (c < tpc_fiducial_volume_endpoints[coord][1]))
        return out
    
    @staticmethod
    def compute_decay_integral(df):
        df['pdark_dir', 'x'] = df['pdark', 'x']/df['p3dark', '']
        df['pdark_dir', 'y'] = df['pdark', 'y']/df['p3dark', '']
        df['pdark_dir', 'z'] = df['pdark', 'z']/df['p3dark', '']
        
        t_x_min = (tpc_fiducial_volume_endpoints[0][0] - df['int_point', 'x'])/df['pdark_dir', 'x']
        t_x_max = (tpc_fiducial_volume_endpoints[0][1] - df['int_point', 'x'])/df['pdark_dir', 'x']
        t_y_min = (tpc_fiducial_volume_endpoints[1][0] - df['int_point', 'y'])/df['pdark_dir', 'y']
        t_y_max = (tpc_fiducial_volume_endpoints[1][1] - df['int_point', 'y'])/df['pdark_dir', 'y']
        
        for tpc_index in [0, 1, 2]:
            t_z_min = (tpc_fiducial_volume_endpoints[2][0] + tpc_index * (tpc_outer_volume[2] + fgd_outer_volume[2]) - df['int_point', 'z'])/df['pdark_dir', 'z']
            t_z_max = (tpc_fiducial_volume_endpoints[2][1] + tpc_index * (tpc_outer_volume[2] + fgd_outer_volume[2]) - df['int_point', 'z'])/df['pdark_dir', 'z']
            t_exit_s = np.stack([t_x_min, t_x_max, t_y_min, t_y_max, t_z_min, t_z_max], axis=1)
            t_exit_s = np.clip(t_exit_s, a_min=0, a_max=None)
            
            t_x_min_in = exp_analysis.is_point_in_tpc_general((df['int_point', 'y'] + t_x_min*df['pdark_dir', 'y'], 
                                                              df['int_point', 'z'] + t_x_min*df['pdark_dir', 'z']),
                                                              coords=[1, 2],
                                                              tpc_index=tpc_index)
            t_x_max_in = exp_analysis.is_point_in_tpc_general((df['int_point', 'y'] + t_x_max*df['pdark_dir', 'y'], 
                                                              df['int_point', 'z'] + t_x_max*df['pdark_dir', 'z']),
                                                              coords=[1, 2],
                                                              tpc_index=tpc_index)
            t_y_min_in = exp_analysis.is_point_in_tpc_general((df['int_point', 'x'] + t_y_min*df['pdark_dir', 'x'], 
                                                              df['int_point', 'z'] + t_y_min*df['pdark_dir', 'z']),
                                                              coords=[0, 2],
                                                              tpc_index=tpc_index)
            t_y_max_in = exp_analysis.is_point_in_tpc_general((df['int_point', 'x'] + t_y_max*df['pdark_dir', 'x'], 
                                                              df['int_point', 'z'] + t_y_max*df['pdark_dir', 'z']),
                                                              coords=[0, 2],
                                                              tpc_index=tpc_index)
            t_z_min_in = exp_analysis.is_point_in_tpc_general((df['int_point', 'x'] + t_z_min*df['pdark_dir', 'x'], 
                                                              df['int_point', 'y'] + t_z_min*df['pdark_dir', 'y']),
                                                              coords=[0, 1],
                                                              tpc_index=tpc_index)
            t_z_max_in = exp_analysis.is_point_in_tpc_general((df['int_point', 'x'] + t_z_max*df['pdark_dir', 'x'], 
                                                              df['int_point', 'y'] + t_z_max*df['pdark_dir', 'y']),
                                                              coords=[0, 1],
                                                              tpc_index=tpc_index)

            are_t_exit_s_in = np.stack([t_x_min_in, t_x_max_in, t_y_min_in, t_y_max_in, t_z_min_in, t_z_max_in], axis=1)
            
            out = np.zeros((len(df), 2))
            mask = are_t_exit_s_in.sum(axis=1) == 2
            aux = t_exit_s[mask][are_t_exit_s_in[mask]]
            out[mask] = aux.reshape((-1, 2))
            df[f'exp_integral_points_{tpc_index*2}'] = out[:, 0]
            df[f'exp_integral_points_{tpc_index*2+1}'] = out[:, 1]

    # @staticmethod
    # def compute_ctau_integral_weights(df, ctau):
    #     ctau = np.asarray(ctau)
    #     scale = np.expand_dims(df['betagamma'], axis=list(range(1, 1+len(ctau.shape))))*\
    #             np.expand_dims(ctau, axis=0)
    #     exp_dims = list(range(1, len(scale.shape)))
    #     out = np.exp(-np.expand_dims(df[f'exp_integral_points_0'], axis=exp_dims)/scale) -\
    #            np.exp(-np.expand_dims(df[f'exp_integral_points_1'], axis=exp_dims)/scale) +\
    #            np.exp(-np.expand_dims(df[f'exp_integral_points_2'], axis=exp_dims)/scale) -\
    #            np.exp(-np.expand_dims(df[f'exp_integral_points_3'], axis=exp_dims)/scale) +\
    #            np.exp(-np.expand_dims(df[f'exp_integral_points_4'], axis=exp_dims)/scale) -\
    #            np.exp(-np.expand_dims(df[f'exp_integral_points_5'], axis=exp_dims)/scale)
    #     return out
    
    @staticmethod
    def compute_ctau_integral_weights(df, betagamma_ctau):
        scale = np.asarray(betagamma_ctau)
        exp_dims = list(range(1, len(scale.shape)))
        out = np.exp(-np.expand_dims(df[f'exp_integral_points_0'], axis=exp_dims)/scale) -\
               np.exp(-np.expand_dims(df[f'exp_integral_points_1'], axis=exp_dims)/scale) +\
               np.exp(-np.expand_dims(df[f'exp_integral_points_2'], axis=exp_dims)/scale) -\
               np.exp(-np.expand_dims(df[f'exp_integral_points_3'], axis=exp_dims)/scale) +\
               np.exp(-np.expand_dims(df[f'exp_integral_points_4'], axis=exp_dims)/scale) -\
               np.exp(-np.expand_dims(df[f'exp_integral_points_5'], axis=exp_dims)/scale)
        return out
        
    @staticmethod
    def unitary_decay_length(df):
        d_decay = np.random.exponential(scale=df['betagamma', '']) # it's for ctau=1
        df['unitary_decay_length', 'x'] = d_decay*df['pdark', 'x']/df['p3dark', '']
        df['unitary_decay_length', 'y'] = d_decay*df['pdark', 'y']/df['p3dark', '']
        df['unitary_decay_length', 'z'] = d_decay*df['pdark', 'z']/df['p3dark', '']
    
    @staticmethod
    def decay_in_tpc(df, ctaus):
        if type(ctaus) is not list:
            ctaus = [ctaus]
        for ctau in ctaus:
            df[f'decay_point_{ctau}_x'.replace('.', '')] = df['int_point_x'] + ctau*df['unitary_decay_length_x']
            df[f'decay_point_{ctau}_y'.replace('.', '')] = df['int_point_y'] + ctau*df['unitary_decay_length_y']
            df[f'decay_point_{ctau}_z'.replace('.', '')] = df['int_point_z'] + ctau*df['unitary_decay_length_z']
            
            df[f'decay_in_tpc_{ctau}'.replace('.', '')] = exp_analysis.is_point_in_tpc(df[f'decay_point_{ctau}_x'.replace('.', '')],
                                                         df[f'decay_point_{ctau}_y'.replace('.', '')],
                                                         df[f'decay_point_{ctau}_z'.replace('.', '')])
    
    @staticmethod
    def decay_in_tpc_fast(int_x, int_y, int_z, length_x, length_y, length_z, ctau):
        decay_x = int_x + ctau*length_x
        decay_y = int_y + ctau*length_y
        decay_z = int_z + ctau*length_z
        
        return exp_analysis.is_point_in_tpc(decay_x, decay_y, decay_z)
    
    def ctau_acceptance(self, ctaus):
        for df in self.dfs.values():
            self.decay_in_tpc(df, ctaus)
        self.decay_in_tpc(self.df_base, ctaus)

    @staticmethod
    def compute_selection(df):
        df['no_cuts', ''] = np.ones(len(df), dtype=bool)
        df['cut1', ''] = (df['ee_beam_costheta', ''] > 0.992)
        df['cut2', ''] = (df['experimental_t', ''] < 0.03)
        df['cut3', ''] = (df['ee_costheta', ''] > 0)
        df['cut4', ''] = (df['ee_momentum', ''] > 0.150)
        df['cut_based', ''] = df['cut1', ''] &\
                            df['cut2', ''] &\
                            df['cut3', ''] &\
                            df['cut4', '']

    @staticmethod
    def kde_on_a_point(df, this_m4mz, distance=default_kde_pars['distance'], smoothing=default_kde_pars['smoothing'], kernel=default_kde_pars['kernel']):
        this_m4mz = np.asarray(this_m4mz)
        this_m4mz_values = np.stack([df['m4'], df['mzprime']], axis=-1)
        return kde_Nd_weights(x=this_m4mz,
                                x_i=this_m4mz_values,
                                smoothing=smoothing,
                                distance=distance,
                                kernel=kernel)

    def no_scan_benchmark_grid(self, function):
        out = []
        for m4 in self.m4_scan:
            out.append([])
            for mz in self.mz_scan:
                if ((self.hierarchy == 'heavy') and (m4 >= mz)) or ((self.hierarchy == 'light') and (m4 <= mz)):
                        out[-1].append(0)
                        continue
                else:
                    out[-1].append(function(self.dfs[(m4, mz)]))
        return np.array(out)
    
    def compute_effective_parameters(self, m4, mz, alpha_dark, epsilon, Umu4_2, Ud4_2):
        aux_grid_m4, aux_grid_mz = np.meshgrid(m4,
                                        mz,
                                        indexing='ij')
        grid_m4 = np.expand_dims(aux_grid_m4, axis=(3, 4, 5, 6))
        grid_mz = np.expand_dims(aux_grid_mz, axis=(3, 4, 5, 6))

        aux_grid_Umu4_2, aux_grid_alpha_dark, aux_grid_epsilon = np.meshgrid(Umu4_2,
                                                               alpha_dark,
                                                               epsilon,
                                                               indexing='ij')
        aux_grid_Vmu4_alpha_epsilon2 = aux_grid_alpha_dark * aux_grid_Umu4_2 * alphaQED * aux_grid_epsilon**2
        grid_Vmu4_alpha_epsilon2 = np.expand_dims(aux_grid_Vmu4_alpha_epsilon2, axis=[0, 1, 2, 6])

        if self.hierarchy == 'light':
            return grid_m4, grid_mz, grid_Vmu4_alpha_epsilon2
        else:
            aux_grid_m4, aux_grid_mz, aux_grid_Ud4_2, aux_grid_alpha_dark, aux_grid_epsilon = np.meshgrid(m4,
                                                                                    mz,
                                                                                    Ud4_2,
                                                                                    alpha_dark,
                                                                                    epsilon,
                                                                                    indexing='ij')
            aux_grid_Vd4_alpha_epsilon2 = aux_grid_alpha_dark * aux_grid_Ud4_2 * alphaQED * aux_grid_epsilon**2
            grid_Vd4_alpha_epsilon2 = np.expand_dims(aux_grid_Vd4_alpha_epsilon2, axis=[0, 1, 2, 5])
            return grid_m4, grid_mz, grid_Vmu4_alpha_epsilon2, grid_Vd4_alpha_epsilon2
        
    def compute_expectation(self, df, m4, mz, alpha_dark, epsilon, Umu4_2, Ud4_2,
                            ntarget_per_material, 
                            pot,
                            selection_query=None,
                            efficiency_factor=1,
                            external_ctau=False,
                            distance=default_kde_pars['distance'], 
                            smoothing=default_kde_pars['smoothing'], 
                            kernel=default_kde_pars['kernel'],
                            out_different_weights=False,
                            no_pot_efficiency=False,
                            is_scan=True):
        # the output is (len(df), len(m4), len(mz), len(alpha_dark), len(epsilon), len(Umu4_2), len(Ud4_2)
        if selection_query is not None:
            aux_df = df.query(selection_query)
        else:
            aux_df = df
        
        # generator weights
        generator_weights = np.expand_dims(aux_df['generator_weight'], axis=list(range(1, 7)))
        
        # upscattering weights
        grid_Umu4_2, grid_alpha_dark, grid_epsilon = np.meshgrid(Umu4_2,
                                                               alpha_dark,
                                                               epsilon,
                                                               indexing='ij')
        Vmu4_alpha_epsilon2 = grid_alpha_dark * grid_Umu4_2 * alphaQED * grid_epsilon**2
        upscattering_weights = Vmu4_alpha_epsilon2 / self.Vmu4_alpha_epsilon2
        upscattering_weights = np.expand_dims(upscattering_weights, axis=[0, 1, 2, 6])
        
        # ctau weights
        if external_ctau is False:
            if self.hierarchy == 'heavy':
                grid_m4, grid_mz, grid_Ud4_2, grid_alpha_dark, grid_epsilon = np.meshgrid(m4,
                                                                                       mz,
                                                                                       Ud4_2,
                                                                                       alpha_dark,
                                                                                       epsilon,
                                                                                       indexing='ij')
                ctaus = ctau_heavy(grid_m4, 
                                   grid_mz, 
                                   grid_alpha_dark * grid_Ud4_2 * alphaQED * grid_epsilon**2, 
                                   self.D_or_M)
                ctaus = np.expand_dims(ctaus, axis=[5])
                extra_m4 = np.expand_dims(np.atleast_1d(m4), axis=[0, 2, 3, 4, 5, 6])
                betagamma_ctaus = np.expand_dims(aux_df['p3dark'], axis=list(range(1, 7)))/extra_m4 * ctaus
                ctau_weights = exp_analysis.compute_ctau_integral_weights(aux_df, betagamma_ctaus)
            elif self.hierarchy == 'light':
                ctaus = None
                ctau_weights = np.ones(len(aux_df))
                ctau_weights = np.expand_dims(ctau_weights, axis=list(range(1, 7)))
        elif external_ctau is None:
            ctaus = None
            ctau_weights = np.ones(len(aux_df))
            ctau_weights = np.expand_dims(ctau_weights, axis=list(range(1, 7)))
        else:
            # ctau weights are in the position of Ud4_2
            ctaus = np.expand_dims(external_ctau, axis=[0, 1, 2, 3, 4, 5])
            extra_m4 = np.expand_dims(np.atleast_1d(m4), axis=[0, 2, 3, 4, 5, 6])
            betagamma_ctaus = np.expand_dims(aux_df['p3dark'], axis=list(range(1, 7)))/extra_m4 * ctaus
            ctau_weights = exp_analysis.compute_ctau_integral_weights(aux_df, betagamma_ctaus)         
        # kde weights
        if is_scan:
            kde_weights = self.kde_on_a_point(aux_df, np.stack(np.meshgrid(m4, mz, indexing='ij'), axis=-1),
                                            distance, smoothing, kernel)
            kde_weights = np.expand_dims(kde_weights, axis=[3, 4, 5, 6])
        else:
            kde_weights = np.ones(len(aux_df))
            kde_weights = np.expand_dims(kde_weights, axis=[1, 2, 3, 4, 5, 6])
        N_kde = np.count_nonzero(kde_weights, axis=0)

        all_weights = generator_weights *\
                          upscattering_weights *\
                          ctau_weights *\
                          kde_weights
        
        if no_pot_efficiency:
            all_weights = np.squeeze(all_weights)
            return aux_df, all_weights, N_kde, ctaus
        else:
            # pot_ntarget weights
            pot_ntarget_weights = self.compute_pot_ntarget_weights(aux_df, ntarget_per_material, pot)
            pot_ntarget_weights = np.expand_dims(pot_ntarget_weights, axis=list(range(1, 7)))

            if out_different_weights:
                return aux_df, (generator_weights, upscattering_weights, ctau_weights, kde_weights, pot_ntarget_weights, efficiency_factor), N_kde, ctaus
            else:
                all_weights *= pot_ntarget_weights * efficiency_factor
                all_weights = np.squeeze(all_weights)
                return aux_df, all_weights, N_kde, ctaus

    def compute_likelihood_from_pars(self, df, m4, mz, alpha_dark, epsilon, Umu4_2, Ud4_2, analysis, external_ctau=False, like_normalized=True, is_scan=True):
        aux_df, all_weights, N_kde, ctaus = self.compute_expectation(df=df,
                                                                     m4=m4,
                                                                     mz=mz,
                                                                     alpha_dark=alpha_dark,
                                                                     epsilon=epsilon,
                                                                     Umu4_2=Umu4_2,
                                                                     Ud4_2=Ud4_2,
                                                                     external_ctau=external_ctau,
                                                                     ntarget_per_material=analysis['n_target'], 
                                                                     pot=analysis['pot'],
                                                                     selection_query=analysis['selection'],
                                                                     efficiency_factor=analysis['efficiency'],
                                                                     is_scan=is_scan)
        return self.compute_likelihood(aux_df, all_weights, analysis, like_normalized)
    
    def compute_likelihood(self, aux_df, all_weights, analysis, like_normalized=True):
        # moveaxis broacast_to etc. could be simplified if we store weights with the number of entries as last dimension
        # possibly output mu, sigma2 and likelihood without summing? with a function that computes the final number?
        all_weights = np.squeeze(all_weights)
        mu_hist = all_weights.sum(axis=0)
        sigma2_hist = (all_weights**2).sum(axis=0)
    
        if analysis['var'] is not None:
            xs = aux_df[analysis['var']].values
            n_entries = len(xs)
            xs = np.broadcast_to(xs, all_weights.T.shape).T
            
            aux_mu = np.concatenate([xs, all_weights], axis=0)
            aux_sigma2 = np.concatenate([xs, all_weights**2], axis=0)
            
            def hist1d(a):
                return np.histogram(a[:n_entries], bins=analysis['binning'], weights=a[n_entries:])[0]
            mu_hist = np.apply_along_axis(func1d=hist1d, 
                                          axis=0, arr=aux_mu)
            mu_hist = np.moveaxis(mu_hist, 0, -1)
            sigma2_hist = np.apply_along_axis(func1d=hist1d, 
                                          axis=0, arr=aux_sigma2)
            sigma2_hist = np.moveaxis(sigma2_hist, 0, -1)

        sigma2_full = sigma2_hist + (analysis['syst']*(analysis['mc']+mu_hist))**2
        leff = LEff_v(analysis['data'], 
                      analysis['mc'] + mu_hist, 
                      sigma2_full)
        
        if analysis['var'] is not None:
            leff = leff.sum(axis=-1)
            
        if like_normalized:
            leff *= -1
            leff -= leff.min()
        
        return leff, mu_hist, sigma2_hist, sigma2_full