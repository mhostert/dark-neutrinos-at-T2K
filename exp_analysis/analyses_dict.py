from copy import deepcopy
import numpy as np

from parameters_dict import *

def create_projected_analysis(analysis, new_pot, new_masses):
    new_analysis = deepcopy(analysis)
    # old_pot = 0
    # for sub_analysis in new_analysis.values():
    #     old_pot += sub_analysis['pot']
    max_mat = max(new_masses, key=new_masses.get)
    for nu_mode, sub_analysis in new_analysis.items():
        old_masses = sub_analysis['masses']
        sub_analysis['masses'] = new_masses
        sub_analysis['scale_factor'] = new_pot[nu_mode]/sub_analysis['pot'] * new_masses[max_mat]/old_masses[max_mat]
        sub_analysis['pot'] = new_pot[nu_mode]
        sub_analysis['mc'] *= sub_analysis['scale_factor']
        sub_analysis['data'] = np.asarray(sub_analysis['mc']).astype(int)
    return new_analysis
        

pot_phase1 = {'FHC': 2.116e21,
              'RHC': 1.65e21}

total_pot_phase1 = pot_phase1['FHC'] + pot_phase1['RHC']
total_pot_phase2 = 20e21 - total_pot_phase1

pot_phase2 = {nu_mode: pot*total_pot_phase2/total_pot_phase1 for nu_mode, pot in pot_phase1.items()}

# analyses
analyses = {}

analyses['tpc'] = {
    'FHC':{
        'pot':12.34e20,
        'syst':0.2,
        'binning':1,
        'var':None,
        'data':0,
        'mc':0.563,
        'masses':tpc_masses,
        'selection':'cut_based',
        'efficiency': 0.1,
    },
    'RHC':{
        'pot':6.29e20,
        'syst':0.2,
        'binning':1,
        'var':None,
        'data':0,
        'mc': 0.015,
        'masses':tpc_masses,
        'selection':'cut_based',
        'efficiency': 0.1,
    }
}

# split TPC into the two production modes
analyses['tpc_pod_only'] = deepcopy(analyses['tpc'])
analyses['tpc_argon_only'] = deepcopy(analyses['tpc'])
for nu_mode in analyses['tpc'].keys():
    analyses['tpc_pod_only'][nu_mode]['selection'] += ' & (~argon)' 
    analyses['tpc_argon_only'][nu_mode]['selection'] += ' & (argon)' 

# phase 1 projection
analyses['tpc_projection1'] = create_projected_analysis(analyses['tpc'],
                                                       pot_phase1,
                                                       tpc_masses)
# phase 2 projection
analyses['tpc_projection2'] = create_projected_analysis(analyses['tpc'],
                                                       pot_phase2,
                                                       tpc_masses_projection_phase2)
### Light case
analyses['tpc_light'] = deepcopy(analyses['tpc'])
for nu_mode in analyses['tpc_light'].keys():
    analyses['tpc_light'][nu_mode]['masses'] = tpc_masses_light_case
    
# phase 1 projection
analyses['tpc_light_projection1'] = create_projected_analysis(analyses['tpc_light'],
                                                       pot_phase1,
                                                       tpc_masses_light_case)
# phase 2 projection
analyses['tpc_light_projection2'] = create_projected_analysis(analyses['tpc_light'],
                                                       pot_phase2,
                                                       tpc_masses_projection_phase2)

analyses['nueccqe_fgd'] = {
    'FHC':{
        'pot': 11.92e20,
        'syst': 0.23,
        'binning': np.linspace(0.00, 0.2, 21),
        'var': 'ee_mass_reco',
        'data': np.loadtxt('../digitized/nueCCQE_ND280_2020/FHC_electron_data.dat')[:, 1].astype(int),
        'mc': np.loadtxt('../digitized/nueCCQE_ND280_2020/FHC_electron_MCtot.dat')[:, 1],
        'masses': fgd1_masses,
        'selection': None,
        'efficiency': 0.1,
    },
    'RHC':{
        'pot': 6.29e20,
        'syst': 0.21,
        'binning': np.linspace(0.00, 0.2, 21),
        'var': 'ee_mass_reco',
        'data': np.loadtxt('../digitized/nueCCQE_ND280_2020/RHC_electron_data.dat')[:, 1].astype(int),
        'mc': np.loadtxt('../digitized/nueCCQE_ND280_2020/RHC_electron_MCtot.dat')[:, 1],
        'masses': fgd1_masses,
        'selection': None,
        'efficiency': 0.1,
    }
}

analyses['single_photon_fgd'] = {
    'FHC':{
        'pot':5.738e20,
        'syst':0.23,
        'binning': np.linspace(0.00, 0.3, 61),
        'var':'ee_mass_reco',
        'data': np.loadtxt('../digitized/T2K_single_photon/FHC_electron_data.dat')[:, 1].astype(int),
        'mc': np.loadtxt('../digitized/T2K_single_photon/FHC_electron_MCtot.dat')[:, 1],
        'masses':fgd1_masses,
        'selection':None,
        'efficiency': 0.1,
    },
}

# phase 1 projection
analyses['nueccqe_fgd_projection1'] = create_projected_analysis(analyses['nueccqe_fgd'],
                                                       pot_phase1,
                                                       fgd_masses_projection_phase1)
analyses['single_photon_fgd_projection1'] = create_projected_analysis(analyses['single_photon_fgd'],
                                                       pot_phase1,
                                                       fgd_masses_projection_phase1)
# phase 2 projection
analyses['nueccqe_fgd_projection2'] = create_projected_analysis(analyses['nueccqe_fgd'],
                                                       pot_phase2,
                                                       fgd_masses_projection_phase2)
analyses['single_photon_fgd_projection2'] = create_projected_analysis(analyses['single_photon_fgd'],
                                                       pot_phase2,
                                                       fgd_masses_projection_phase2)

# adding ntargets

mol2natoms = 6.02214e23 # Avogadro's number
ton2grams = 1e6

for analysis in analyses.values():
    for sub_analysis in analysis.values():
        sub_analysis['n_target'] = {mat:mass*ton2grams/molar_mass[mat]*mol2natoms
                                     for mat, mass in sub_analysis['masses'].items()}