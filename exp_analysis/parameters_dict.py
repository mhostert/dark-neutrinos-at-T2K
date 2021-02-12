physics_parameters = {}

physics_parameters['heavy'] = {
    'm4_limits': (0.005, 1),
    'mz_limits': (0.005, 10),
    'm4_scan' : [0.01, 0.1, 0.5],
    'mz_scan': [0.01, 0.1, 1, 5],
    'alpha_dark': 0.4,
    'Umu4': 2.2e-7,
    'espilon2': 4.6e-4,
    'alpha_em': 1./137,
}

# physics_parameters['heavy']['span_2d'] = (physics_parameters['heavy']['m4_limits'][1] - physics_parameters['heavy']['m4_limits'][0]) *\
#                                          (physics_parameters['heavy']['mz_limits'][1] - physics_parameters['heavy']['mz_limits'][0])

physics_parameters['heavy']['span_2d'] = (physics_parameters['heavy']['mz_limits'][1] - physics_parameters['heavy']['mz_limits'][0] +\
                                          physics_parameters['heavy']['mz_limits'][1] - physics_parameters['heavy']['m4_limits'][0]) *\
                                         (physics_parameters['heavy']['m4_limits'][1] - physics_parameters['heavy']['m4_limits'][0])/2

physics_parameters['heavy']['Vmu4_alpha_epsilon2'] = physics_parameters['heavy']['alpha_dark'] *\
                                                     physics_parameters['heavy']['Umu4'] *\
                                                     physics_parameters['heavy']['alpha_em'] *\
                                                     physics_parameters['heavy']['espilon2']

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