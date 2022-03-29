import numpy as np

molar_mass = {
    'hydrogen': 1.00784,
    'carbon': 12.0107,
    'oxygen': 15.999,
    'copper': 63.546,
    'zinc': 65.38,
    'lead': 207.2,
    # 'argon': 39.948
}
mol2natoms = 6.02214e23 # Avogadro's number
ton2grams = 1e6

tpc_masses = {
    'hydrogen': 3.3*2*molar_mass['hydrogen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen']),
    'oxygen': 3.3*molar_mass['oxygen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen']),
    'carbon': 8.221,
    'copper': 1.315*0.66, # percentage of copper in typical brass
    'zinc': 1.315*0.34, # percentage of zinc in typical brass
    'lead': 3.634,
    # 'argon': 0.01
}

fgd_masses = {'hydrogen': 0,
            'carbon': 0.9195}

analyses = {}

analyses['tpc'] = {
    'FHC':{
        'pot':12.34e20,
        'syst':0.2,
        'binning':1,
        'var':None,
        'data':0,
        'mc':0,
        'masses':tpc_masses,
        # 'selection':'cut_based & (~argon)',
                'selection':'cut_based',
        'efficiency': 0.1,
    },
    'RHC':{
        'pot':6.29e20,
        'syst':0.2,
        'binning':1,
        'var':None,
        'data':0,
        'mc':0,
        'masses':tpc_masses,
        # 'selection':'cut_based & (~argon)',
                'selection':'cut_based',
        'efficiency': 0.1,
    }
}

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

for analysis in analyses.values():
    for sub_analysis in analysis.values():
        sub_analysis['n_target'] = {mat:mass*ton2grams/molar_mass[mat]*mol2natoms
                                     for mat, mass in sub_analysis['masses'].items()}