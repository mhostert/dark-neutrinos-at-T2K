import itertools
import numpy as np
from scipy.interpolate import interpn
import pandas as pd

from sklearn.neighbors import BallTree
from kde_utils import kde_Nd_weights, log_distance

from fourvec import *
from parameters_dict import *
from const import *
from ctau_utils import *

droplist = ['plm_t', 'plm_x', 'plm_y', 'plm_z', 'plp_t', 'plp_x', 'plp_y', 'plp_z',
       'pnu_t', 'pnu_x', 'pnu_y', 'pnu_z', 'pHad_t', 'pHad_x', 'pHad_y',
       'pHad_z', 'weight_decay', 'regime', 'pee_t',
       'pdark_t', 'pee_x', 'pdark_x', 'pee_y', 'pdark_y', 'pee_z', 'pdark_z',
       'recoil_mass', 'p3dark']

class exp_analysis(object):
    
    def __init__(self, hierarchy, D_or_M, base_folder='../data/nd280_nu/3plus1/'):
        self.__dict__.update(physics_parameters[hierarchy])
        self.hierarchy = hierarchy
        self.D_or_M = D_or_M
        self.base_folder = base_folder
        self.dfs = {}
        
    def load_df_base(self, n_evt=1000000, filename=None, build_ball_tree=False, distance='log'):
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
        self.compute_actual_weights(df, which_scan)
        self.compute_interaction_point(df)
        self.unitary_decay_length(df)
        self.compute_selection(df)
        self.compute_decay_point(df)
        
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
            self.actual_weight_values = self.df_base['actual_weight'].values

    def setResolution(self, smearing_p, p_binning, smearing_theta, theta_binning):
        # needs to deal with overflow and underflow bins - TODO
        
        # normalise smearing_matrices so that each column sums to 1
        self.smearing_p = (smearing_p.T/smearing_p.sum(axis=1)).T
        self.smearing_theta = (smearing_theta.T/smearing_theta.sum(axis=1)).T
        
        self.df_base['em_beam_theta_reco'] = true2reco(self.df_base['em_beam_theta'], 
                                                     smearing_theta, theta_binning, theta_binning)
        self.df_base['ep_beam_theta_reco'] = true2reco(self.df_base['ep_beam_theta'], 
                                                     smearing_theta, theta_binning, theta_binning)
        self.df_base['em_energy_reco'] = true2reco(self.df_base['em_energy'], 
                                                     smearing_p, p_binning, p_binning)
        self.df_base['ep_energy_reco'] = true2reco(self.df_base['ep_energy'], 
                                                     smearing_p, p_binning, p_binning)
        
        emp_opening_cos_angle = np.sin(self.df_base['em_beam_theta_reco'])*np.cos(self.df_base['em_beam_phi']) *\
                                np.sin(self.df_base['ep_beam_theta_reco'])*np.cos(self.df_base['ep_beam_phi']) +\
                                np.sin(self.df_base['em_beam_theta_reco'])*np.sin(self.df_base['em_beam_phi']) *\
                                np.sin(self.df_base['ep_beam_theta_reco'])*np.sin(self.df_base['ep_beam_phi']) +\
                                np.cos(self.df_base['em_beam_theta_reco'])*np.cos(self.df_base['ep_beam_theta_reco'])
        
        self.df_base['ee_mass_reco'] = 2*self.df_base['em_energy_reco']*self.df_base['ep_energy_reco']*\
                                        (1 - emp_opening_cos_angle)
        
    @staticmethod
    def true2reco(true_entries, smearing_matrix, binning_true, binning_reco):
        # still needs to deal with overflow and underflow bins
        np.testing.assert_allclose(smearing_matrix.sum(axis=1), np.ones(smearing_matrix.shape[1]))
        smearing_matrix_cumsum = smearing_matrix.cumsum(axis=1)
        aux_p = np.random.rand(len(true_entries))
        true_entries_digitized = np.digitize(true_entries, binning_true)
        reco_entries_digitized = (smearing_matrix_cumsum[true_entries_digitized] < aux_p[:.None]).sum(axis=1)
        return binning_reco[reco_entries_digitized]
        
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
        
        # e+ vars
        df['ep_energy', ''] = df['plp', 't']
        df['ep_beam_costheta', ''] = df['plp', 'z']/np.sqrt(dot3_df(df['plp'], df['plp']))
        df['ep_beam_theta', ''] = np.arccos(df['ep_beam_costheta', ''])
        
        # high level vars
        df['experimental_t', ''] = (df['plm','t'] - df['plm','z'] + df['plp','t'] - df['plp','z'])**2 +\
                                    df['plm','x']**2 + df['plm','y']**2 + df['plp','x']**2 + df['plp','y']**2
        
        df['p3dark', ''] = np.sqrt(dot3_df(df['pdark'], df['pdark']))
        df['mdark', ''] = inv_mass(df['pdark'])
        df['betagamma', ''] = df['p3dark', '']/df['mdark', '']

    def compute_actual_weights(self, df, which_scan=None, with_decay_formula=True, smooth_pars_decaywidth=[0.005, 0.05], kernel='epa', n_points_decaywidth_interpolation=30):
        m4_values = df['m4', ''].values
        mz_values = df['mzprime', ''].values
        weight_values = df['weight', ''].values
        
        if with_decay_formula:
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
        else:
            weight_decay_values = df['weight_decay', ''].values
            if which_scan == None:
                df['total_decay_rate', ''] = weight_decay_values.sum()
            elif which_scan == 'm4_mz':
                m4_span = np.linspace(m4_values.min(), m4_values.max(), int(n_points_decaywidth_interpolation/4))
                mz_span = np.linspace(mz_values.min(), mz_values.max(), int(n_points_decaywidth_interpolation/4))
                aux_grid = np.stack(np.meshgrid(m4_span, mz_span, indexing='ij'), axis=-1)
                aux_values = np.stack([m4_values, mz_values], axis=-1)
                gamma_kde_weights = kde_Nd_weights(aux_grid, 
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
    def is_point_in_tpc(x, y, z):
        return ((((p0d_length - 0.45) < z) & (z < (p0d_length + tpc_length))) |
                ((p0d_length + tpc_length + fgd_length < z) & (z < (p0d_length + tpc_length + fgd_length + tpc_length))) |
                ((p0d_length + 2*(tpc_length + fgd_length) < z) & (z < (p0d_length + 2*(tpc_length + fgd_length) + tpc_length)))) &\
                ((0 < x) & (x < p0d_dimensions[0])) &\
                ((0 < y) & (y < p0d_dimensions[1]))
        
    @staticmethod
    def compute_decay_point(df):
        df['pdark_dir', 'x'] = df['pdark', 'x']/df['p3dark', '']
        df['pdark_dir', 'y'] = df['pdark', 'y']/df['p3dark', '']
        df['pdark_dir', 'z'] = df['pdark', 'z']/df['p3dark', '']
        
        t_0_0 = (p0d_length - df['int_point', 'z'])/df['pdark_dir', 'z']
        t_0_1 = (p0d_length + tpc_length - df['int_point', 'z'])/df['pdark_dir', 'z']
        
        t_1_0 = (p0d_length + tpc_length + fgd_length - df['int_point', 'z'])/df['pdark_dir', 'z']
        x_1_0 = df['int_point', 'x'] + t_1_0 * df['pdark_dir', 'x']
        y_1_0 = df['int_point', 'y'] + t_1_0 * df['pdark_dir', 'y']
        in_tpc_1 = (x_1_0 > 0) & (x_1_0 < p0d_dimensions[0]) &\
                   (y_1_0 > 0) & (y_1_0 < p0d_dimensions[1])
        
        t_2_0 = (p0d_length + 2*(tpc_length + fgd_length) - df['int_point', 'z'])/df['pdark_dir', 'z']
        x_2_0 = df['int_point', 'x'] + t_2_0 * df['pdark_dir', 'x']
        y_2_0 = df['int_point', 'y'] + t_2_0 * df['pdark_dir', 'y']
        in_tpc_2 = (x_2_0 > 0) & (x_2_0 < p0d_dimensions[0]) &\
                   (y_2_0 > 0) & (y_2_0 < p0d_dimensions[1])
        
        how_many_tpcs = np.ones(len(df))
        which_tpc = np.zeros(len(df))
        which_tpc[in_tpc_2] = np.random.choice([0, 1, 2], in_tpc_2.sum())
        how_many_tpcs[in_tpc_2] = 3
        in_tpc_1_but_not_2 = in_tpc_1 & ~in_tpc_2
        which_tpc[in_tpc_1_but_not_2] = np.random.choice([0, 1, 2], in_tpc_1_but_not_2.sum())
        how_many_tpcs[in_tpc_1_but_not_2] = 2
        
        df[f'decay_point', 'z'] = p0d_length + which_tpc*(tpc_length + fgd_length) + np.random.uniform(0, tpc_length, len(df))
        df[f'd_decay'] = (df[f'decay_point', 'z'] - df['int_point', 'z'])/df['pdark_dir', 'z']
        df[f'decay_point', 'x'] = df['int_point', 'x'] + df[f'd_decay'] * df['pdark_dir', 'x']
        df[f'decay_point', 'y'] = df['int_point', 'y'] + df[f'd_decay'] * df['pdark_dir', 'y']
        df[f'decay_in_tpc'] = exp_analysis.is_point_in_tpc(df['decay_point', 'x'],
                                                           df['decay_point', 'y'],
                                                           df['decay_point', 'z'])
        df[f'importance_weight'] = (how_many_tpcs)*(tpc_length/df['pdark_dir', 'z'])
    
    @staticmethod
    def compute_ctau_weights(df, ctau):
        scale = df['betagamma']*ctau
        return 1/scale * np.exp(-df[f'd_decay']/scale) * df[f'importance_weight']
    
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
        out = []
        for ctau in ctaus:
            df[f'decay_point_{ctau}_x'] = df['int_point_x'] + ctau*df['unitary_decay_length_x']
            df[f'decay_point_{ctau}_y'] = df['int_point_y'] + ctau*df['unitary_decay_length_y']
            df[f'decay_point_{ctau}_z'] = df['int_point_z'] + ctau*df['unitary_decay_length_z']
            
            df[f'decay_in_tpc_{ctau}'] = is_point_in_tpc(df[f'decay_point_{ctau}_x'],
                                                         df[f'decay_point_{ctau}_y'],
                                                         df[f'decay_point_{ctau}_z'])
    
    @staticmethod
    def decay_in_tpc_fast(int_x, int_y, int_z, length_x, length_y, length_z, ctau):
        decay_x = int_x + ctau*length_x
        decay_y = int_y + ctau*length_y
        decay_z = int_z + ctau*length_z
        return ((
            ((p0d_length - 0.45) < decay_z) & (decay_z < (p0d_length + tpc_length)) |
            (p0d_length + tpc_length + fgd_length < decay_z) & (decay_z < (p0d_length + tpc_length + fgd_length + tpc_length)) |
            (p0d_length + 2*(tpc_length + fgd_length) < decay_z) & (decay_z < (p0d_length + 2*(tpc_length + fgd_length) + tpc_length)))) &\
            (0 < decay_x) & (decay_x < p0d_dimensions[0]) &\
            (0 < decay_y) & (decay_y < p0d_dimensions[1])
    
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
    def kde_on_a_point(df, this_m4mz, distance='log', smoothing=[0.1, 0.1], kernel='epa'):
        this_m4mz = np.asarray(this_m4mz)
        this_m4mz_values = np.stack([df['m4'], df['mzprime']], axis=-1)
        return df['actual_weight'].values * kde_Nd_weights(x=this_m4mz,
                                                                        x_i=this_m4mz_values,
                                                                        smoothing=smoothing,
                                                                        distance=distance,
                                                                        kernel=kernel)

    @staticmethod
    def kde_on_a_grid(df, m4_scan, mz_scan, distance='log', smoothing=[0.1, 0.1], kernel='epa'):
        grid_to_eval = np.stack(np.meshgrid(m4_scan, mz_scan, indexing='ij'), axis=-1)
        this_m4mz_values = np.stack([df['m4'], df['mzprime']], axis=-1)
        this_kde_weights = kde_Nd_weights(x=grid_to_eval,
                                               x_i=this_m4mz_values,
                                               smoothing=smoothing,
                                               distance=distance,
                                               kernel=kernel)
        return df['actual_weight'].values[:, np.newaxis, np.newaxis] * this_kde_weights

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

    @staticmethod
    def kde_n_events(df, m4mz=None, ctau=None, mu=1, selection_query=None, smoothing=[0.1, 0.1], distance='log', kernel='epa', provide_n_samples=False):
        if ctau is not None:
            ctau_mask = exp_analysis.decay_in_tpc_fast(df['int_point_x'],
                                                    df['int_point_y'],
                                                    df['int_point_z'],
                                                    df['unitary_decay_length_x'],
                                                    df['unitary_decay_length_y'],
                                                    df['unitary_decay_length_z'],
                                                    ctau)
            aux_df = df[ctau_mask]
            if provide_n_samples:
                N_ctau = ctau_mask.sum()
        else:
            aux_df = df
        if selection_query is not None:
            aux_df = aux_df.query(selection_query)   
        if m4mz is not None:
            kde_weights = mu * exp_analysis.kde_on_a_point(df=aux_df, 
                                              this_m4mz=m4mz, 
                                              smoothing=smoothing,
                                              distance=distance,
                                              kernel=kernel)
        else:
            kde_weights = mu * aux_df['actual_weight'].values
        if not provide_n_samples:
            return kde_weights.sum(), (kde_weights**2).sum()
        else:
            N_kde = np.count_nonzero(kde_weights)
            return kde_weights.sum(), (kde_weights**2).sum(), N_ctau, N_kde
    
    @staticmethod
    def kde_n_events_new(df, m4mz=None, ctau=None, mu=1, selection_query=None, smoothing=[0.1, 0.1], distance='log', kernel='epa', provide_n_samples=False):
        if ctau is not None:
            ctau_mask = df['decay_in_tpc']
            if provide_n_samples:
                N_ctau = ctau_mask.sum()
            aux_df = df[ctau_mask]
        else:
            aux_df = df
        if selection_query is not None:
            aux_df = aux_df.query(selection_query)
        
        if ctau is not None:
            ctau_weights = exp_analysis.compute_ctau_weights(aux_df, ctau)
    
        if m4mz is not None:
            kde_weights = mu * exp_analysis.kde_on_a_point(df=aux_df, 
                                              this_m4mz=m4mz, 
                                              smoothing=smoothing,
                                              distance=distance,
                                              kernel=kernel)
        else:
            kde_weights = mu * aux_df['actual_weight'].values
        kde_weights *= ctau_weights
        if not provide_n_samples:
            return kde_weights.sum(), (kde_weights**2).sum()
        else:
            N_kde = np.count_nonzero(kde_weights)
            return kde_weights.sum(), (kde_weights**2).sum(), N_ctau, N_kde

    def kde_n_events_benchmark_grid(self, ctau=None, mu=1, selection_query=None, smoothing=[0.1, 0.1], distance='log', kernel='epa'):
        out = []
        for m4 in self.m4_scan:
            out.append([])
            for mz in self.mz_scan:
                if ((self.hierarchy == 'heavy') and (m4 >= mz)) or ((self.hierarchy == 'light') and (m4 <= mz)):
                        out[-1].append([0, 0])
                        continue
                else:
                    out[-1].append(self.kde_n_events(self.df_base,
                                                     m4mz=(m4, mz),
                                                     ctau=ctau,
                                                     mu=mu,
                                                     selection_query=selection_query,
                                                     smoothing=smoothing,
                                                     distance=distance,
                                                     kernel=kernel))
        return np.array(out)