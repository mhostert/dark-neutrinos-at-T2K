from copy import deepcopy
import numpy as np

from parameters_dict import *

def create_projected_analysis(analysis, new_pot, new_masses):
    new_analysis = deepcopy(analysis)
    old_pot = 0
    for sub_analysis in new_analysis.values():
        old_pot += sub_analysis['pot']
    max_mat = max(new_masses, key=new_masses.get)
    for sub_analysis in new_analysis.values():
        old_masses = sub_analysis['masses']
        sub_analysis['pot'] = new_pot
        sub_analysis['masses'] = new_masses
        sub_analysis['mc'] *= (new_pot/old_pot * new_masses[max_mat]/old_masses[max_mat])
        sub_analysis['data'] = np.asarray(sub_analysis['mc']).astype(int)
    return new_analysis
        
molar_mass = {
    'hydrogen': 1.00784,
    'carbon': 12.0107,
    'oxygen': 15.999,
    'copper': 63.546,
    'zinc': 65.38,
    'lead': 207.2,
    'argon': 39.948
}
mol2natoms = 6.02214e23 # Avogadro's number
ton2grams = 1e6

total_pot_phase1 = 4e21
total_pot_phase2 = 16e21

# tpc masses
tpc_masses = {
    'hydrogen': 3.3*2*molar_mass['hydrogen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen']),
    'oxygen': 3.3*molar_mass['oxygen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen']),
    'carbon': 8.221,
    'copper': 1.315*0.66, # percentage of copper in typical brass
    'zinc': 1.315*0.34, # percentage of zinc in typical brass
    'lead': 3.634, # it seems 3.35 should be a better number
    'argon': 0.016020 # 3 tpc total argon - needs to be reduced for light case and for projections
}

tpc_masses_light_case = {'argon': 0.01}

new_tpc_volume = 2 *\
                 (200 - 2*tpc_fiducial_volume_gap_x) *\
                 (80 - 2*tpc_fiducial_volume_gap_y) *\
                 (180 - tpc_fiducial_volume_gap_z_begin - tpc_fiducial_volume_gap_z_end)

tpc_masses_projection_phase2 = {
    'argon': tpc_masses_light_case['argon'] * (1 + new_tpc_volume/(3*tpc_fiducial_volume))
}

# fgd masses
fgd_masses = {'hydrogen': 0, # this needs to be fixed too!
              'carbon': 0.9195, # 1848.6 * 1e-9 * 184**2  * 15
              'argon': 0
             }

number_fgd = 2

fgd_masses_projection_phase1 = deepcopy(fgd_masses)
for mass in fgd_masses_projection_phase1.values():
    mass *= number_fgd

fgd1_fid_volume = 174.9 * 174.9 * 14.4
fgd1_fid_volume_gap = 5.75
super_fgd_volume = (192 - fgd1_fid_volume_gap) * (192 - fgd1_fid_volume_gap) * 56
fgd_projection_volume_scale_factor_phase2 = super_fgd_volume/fgd1_fid_volume

fgd_masses_projection_phase2 = deepcopy(fgd_masses)
for mass in fgd_masses_projection_phase2.values():
    mass *= (number_fgd + fgd_projection_volume_scale_factor_phase2)

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
                                                       total_pot_phase1,
                                                       tpc_masses)
# phase 2 projection
analyses['tpc_projection2'] = create_projected_analysis(analyses['tpc'],
                                                       total_pot_phase2,
                                                       tpc_masses_projection_phase2)
### Light case
analyses['tpc_light'] = deepcopy(analyses['tpc'])
for nu_mode in analyses['tpc_light'].keys():
    analyses['tpc_light'][nu_mode]['masses'] = tpc_masses_light_case
    
# phase 1 projection
analyses['tpc_light_projection1'] = create_projected_analysis(analyses['tpc_light'],
                                                       total_pot_phase1,
                                                       tpc_masses_light_case)
# phase 2 projection
analyses['tpc_light_projection2'] = create_projected_analysis(analyses['tpc_light'],
                                                       total_pot_phase2,
                                                       tpc_masses_projection_phase2)

analyses['nueccqe_fgd'] = {
    'FHC':{
        'pot': 11.92e20,
        'syst': 0.23,
        'binning': np.linspace(0.00, 0.2, 21),
        'var': 'ee_mass_reco',
        'data': np.loadtxt('../digitized/nueCCQE_ND280_2020/FHC_electron_data.dat')[:, 1].astype(int),
        'mc': np.loadtxt('../digitized/nueCCQE_ND280_2020/FHC_electron_MCtot.dat')[:, 1],
        'masses': fgd_masses,
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
        'masses': fgd_masses,
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
        'masses':fgd_masses,
        'selection':None,
        'efficiency': 0.1,
    },
}

# phase 1 projection
analyses['nueccqe_fgd_projection1'] = create_projected_analysis(analyses['nueccqe_fgd'],
                                                       total_pot_phase1,
                                                       fgd_masses_projection_phase1)
analyses['single_photon_fgd_projection1'] = create_projected_analysis(analyses['single_photon_fgd'],
                                                       total_pot_phase1,
                                                       fgd_masses_projection_phase1)
# phase 2 projection
analyses['nueccqe_fgd_projection2'] = create_projected_analysis(analyses['nueccqe_fgd'],
                                                       total_pot_phase2,
                                                       fgd_masses_projection_phase2)
analyses['single_photon_fgd_projection2'] = create_projected_analysis(analyses['single_photon_fgd'],
                                                       total_pot_phase2,
                                                       fgd_masses_projection_phase2)

# adding ntargets
for analysis in analyses.values():
    for sub_analysis in analysis.values():
        sub_analysis['n_target'] = {mat:mass*ton2grams/molar_mass[mat]*mol2natoms
                                     for mat, mass in sub_analysis['masses'].items()}