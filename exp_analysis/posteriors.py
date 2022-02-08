import numpy as np

from const import alphaQED
from ctau_utils import ctau_heavy
import parameters_dict
from Likelihood import LEff

likelihood_levels_2d = {0.68: -2.3/2,
          0.9: -4.61/2,
          0.95: -5.99/2}

def load_mc_data(sideband, base_folder_digitized='../digitized/nueCCQE_ND280_2020/'):
    digitized_data = np.loadtxt(base_folder_digitized + sideband + '_electron_data.dat')
    digitized_data[:, 0] = parameters_dict.fgd_bin_centers
    digitized_data[:, 1] = digitized_data[:, 1].astype(int)

    digitized_MCtot = np.loadtxt(base_folder_digitized + sideband + '_electron_MCtot.dat')
    digitized_MCtot[:, 0] = parameters_dict.fgd_bin_centers
    return digitized_data, digitized_MCtot

digitized_data = {}
digitized_MCtot = {}
for sideband in ['FHC','RHC']:
    data, MCtot = load_mc_data(sideband)
    digitized_data[sideband] = data    
    digitized_MCtot[sideband] = MCtot
    
def heavy_nonminimal_posterior(my_exp_analyses, 
                               fluxes=['FHC', 'RHC'], 
                               hierarchy='heavy',
                               D_or_M='majorana',
                               m4=0.1, mz=1.25, Umu4_2=2.2e-7, Ud4_2=1, alpha_d=0.4, epsilon=2.1e-2,
                               systs=parameters_dict.tpc_systematic_uncertainties, 
                               additional_efficiency=0.1,
                               selection_query='cut_based',
                               ctau_mode='integral',
                               distance='log',
                               smoothing_pars=(0.1, 0.1)):
    Vmu4_alpha_epsilon2 = alpha_d * Umu4_2 * alphaQED * epsilon**2
    Valpha4_alpha_epsilon2 = alpha_d * Ud4_2 * alphaQED * epsilon**2
    ctau = ctau_heavy(m4, 
                      mz, 
                      Valpha4_alpha_epsilon2, 
                      D_or_M)

    leff, mu, sigma2, N_ctau, N_final = 0, 0, 0, 0, 0
    for flux in fluxes:
        my_exp_analysis = my_exp_analyses[f'{hierarchy}_{D_or_M}_{flux}']
        this_mu = Vmu4_alpha_epsilon2/my_exp_analysis.Vmu4_alpha_epsilon2 * additional_efficiency
        syst = systs[flux]
        aux_mu, aux_sigma2, aux_N_ctau, aux_N_final = my_exp_analysis.kde_n_events(df=my_exp_analysis.df_base,
                                                                                   selection_query=selection_query,
                                                                                     m4mz=(m4, mz),
                                                                                     ctau=ctau,
                                                                                     mu=this_mu,
                                                                                     distance=distance,
                                                                                     smoothing=smoothing_pars,
                                                                                     provide_n_samples=True,
                                                                                     ctau_mode=ctau_mode)
        leff += LEff(0, aux_mu, aux_sigma2 + (syst*aux_mu)**2)
        mu += aux_mu
        sigma2 += (aux_sigma2 + (syst*aux_mu)**2) 
        N_ctau += aux_N_ctau
        N_final += aux_N_final
    return leff, mu, sigma2, ctau, N_ctau, N_final

heavy_nonminimal_posterior_v = np.vectorize(heavy_nonminimal_posterior, excluded=['my_exp_analyses', 'fluxes', 'hierarchy', 'D_or_M',
                                                                                  'method', 'systs', 'additional_efficiency', 
                                                                                  'ctau_mode', 'selection_query'])


def light_minimal_posterior(my_exp_analyses, 
                               fluxes=['FHC', 'RHC'], 
                               hierarchy='light', 
                               D_or_M='majorana',
                               m4=0.1, mz=0.03, Umu4_2=8e-9, alpha_d=0.25, epsilon=1.7e-4,
                               systs=parameters_dict.fgd_systematic_uncertainties, 
                               selection_query='carbon',
                               additional_efficiency=parameters_dict.ratio_fgd_mass_p0d_carbon*parameters_dict.fgd_efficiency,
                               signal_var='ee_mass_reco',
                               n_observed=digitized_data, 
                               n_predicted_sm=digitized_MCtot, 
                               binning=parameters_dict.fgd_binning,
                               sensitivity=False,
                               sensitivity_scale_factor=parameters_dict.fgd_sensitivity_scale_factor,
                               distance='log',
                               smoothing_pars=(0.1, 0.1)):
    Vmu4_alpha_epsilon2 = alpha_d * Umu4_2 * alphaQED * epsilon**2
    
    leff, mu, sigma2, = 0, 0, 0
    for flux in fluxes:
        my_exp_analysis = my_exp_analyses[f'{hierarchy}_{D_or_M}_{flux}']
        this_mu = Vmu4_alpha_epsilon2/my_exp_analysis.Vmu4_alpha_epsilon2 * additional_efficiency
        syst = systs[flux]

        kde_weights, aux_df = my_exp_analysis.kde_n_events(df=my_exp_analysis.df_base.query(selection_query),
                                                     m4mz=(m4, mz),
                                                     ctau=None,
                                                     mu=this_mu,
                                                     distance=distance,
                                                     smoothing=smoothing_pars,
                                                     return_df=True)

        kde_weights, aux_df['ee_mass'], aux_df['ee_mass_reco']

        if signal_var == 'ee_mass_no_kde':
            this_var = mz*np.ones(len(aux_df))
        else:
            this_var = aux_df[signal_var]

        mu_hist, _ = np.histogram(this_var, weights=kde_weights, bins=binning)
        sigma2_hist, _ = np.histogram(this_var, weights=kde_weights**2, bins=binning)
        
        if sensitivity:
            this_mc = n_predicted_sm[flux][:, 1] * sensitivity_scale_factor
            this_data = this_mc.astype(int)
            mu_hist *= sensitivity_scale_factor
            sigma2_hist *= sensitivity_scale_factor**2
        else:
            this_mc = n_predicted_sm[flux][:, 1]
            this_data = n_observed[flux][:, 1]

        for obs, pred, aux_mu, aux_sigma2 in zip(this_data, this_mc, mu_hist, sigma2_hist):
            leff += LEff(obs, pred+aux_mu, aux_sigma2 + (syst*(pred+aux_mu))**2)
            mu += aux_mu
            sigma2 += (aux_sigma2 + (syst*(pred+aux_mu))**2)
        
    return leff, mu, sigma2

light_minimal_posterior_v = np.vectorize(light_minimal_posterior, excluded=['my_exp_analyses', 
                                                                           'fluxes',
                                                                           'hierarchy', 
                                                                           'D_or_M',
                                                                           'systs', 
                                                                           'selection_query',
                                                                           'additional_efficiency',
                                                                           'n_observed', 
                                                                           'n_predicted_sm', 
                                                                           'binning',
                                                                           'sensitivity',
                                                                           'sensitivity_scale_factor',
                                                                           'distance',
                                                                           'smoothing_pars'])