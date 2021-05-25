import numpy as np
from const import alphaQED

upper_bound_epsilon = 0.05 # model independent constraint on epsilon - probably we can push to 3.5%
upper_bound_Umu4_2 = 0.03 # 

physics_parameters = {}

physics_parameters['heavy'] = {
    'm4_limits': (0.005, 1),
    'mz_limits': (0.005, 10),
    'm4_scan' : [0.01, 0.1, 0.5],
    'mz_scan': [0.02, 0.2, 1, 5],
    'alpha_dark': 0.4,
    'Umu4': 2.2e-7,
    'epsilon2': 4.6e-4,
    'lower_bound_Vmu4_alpha_epsilon2': 10**(-17),
}

physics_parameters['light'] = {
    'm4_limits': (0.005, 2),
    'mz_limits': (0.005, 2),
    'm4_scan' : [0.01, 0.1, 0.4, 0.8],
    'mz_scan': [0.0075, 0.03, 0.1, 0.4],
    'alpha_dark': 0.4,
    'Umu4': 2e-9,
    'epsilon2': 2e-10/alphaQED,
    # 'lower_bound_Vmu4_alpha_epsilon2': 10**(-21),
    'lower_bound_Vmu4': 10**(-10),
    'lower_bound_epsilon': 10**(-5),
}

for physics_params in physics_parameters.values():
    physics_params['upper_bound_epsilon'] = upper_bound_epsilon
    physics_params['upper_bound_Vmu4'] = upper_bound_Umu4_2
    physics_params['Vmu4_alpha_epsilon2'] = physics_params['alpha_dark'] *\
                                                         physics_params['Umu4'] *\
                                                         alphaQED *\
                                                         physics_params['epsilon2'] 
    physics_params['upper_bound_Vmu4_alpha_epsilon2'] = physics_params['upper_bound_Vmu4'] *\
                                                        alphaQED *\
                                                        upper_bound_epsilon**2
    physics_params['upper_bound_Valpha4_alpha_epsilon2'] = alphaQED *\
                                                        upper_bound_epsilon**2


    # physics_params['upper_bound_log10_Vmu4_alpha_epsilon2'] = np.log10(physics_params['upper_bound_Vmu4_alpha_epsilon2'])
    # physics_params['lower_bound_log10_Vmu4_alpha_epsilon2'] = np.log10(physics_params['lower_bound_Vmu4_alpha_epsilon2'])
    # physics_params['upper_bound_log10_Valpha4_alpha_epsilon2'] = np.log10(physics_params['upper_bound_Valpha4_alpha_epsilon2'])

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

cuts_dict = {
    'cut1' : r'cos $\theta_{ee, beam}$ > 0.99',
    'cut2' : r'$t_{exp}$ < 0.03',
    'cut3' : r'cos $\theta_{ee}$ > 0',
    'cut4' : r'$p_{ee}$ > 0.15',
}