import numpy as np
from const import alphaQED

likelihood_levels_2d = {0.68: 2.3/2,
          0.9: 4.61/2,
          0.95: 5.99/2}

upper_bound_epsilon = 0.05 # model independent constraint on epsilon - probably we can push to 3.5%
upper_bound_Umu4_2 = 0.03 # 

physics_parameters = {}

physics_parameters['heavy'] = {
    'm4_limits': (0.005, 1),
    'mz_limits': (0.005, 10),
    'm4_scan' : [0.01, 0.1, 0.5],
    'mz_scan': [0.02, 0.2, 1, 5],
    'bp': {
        # 'm4':0.12,
        'm4':0.1,
        'mz':1.25,
        'alpha_dark':0.4,
        'epsilon':np.sqrt(4.6e-4),
        'Umu4_2':2.2e-7,
        'Ud4_2':1,
    },
    'ticks': {
        'Ud4_2': np.array([1]),
        'mz': np.geomspace(0.13, 10, 20),
        'epsilon': np.geomspace(1e-5, 1e-1, 25),
        'm4': np.geomspace(0.005, 1, 20),
        'Umu4_2': np.geomspace(1e-11, 1, 25),
    }
    # 'lower_bound_Vmu4_alpha_epsilon2': 10**(-17),
}
physics_parameters['light'] = {
    'm4_limits': (0.005, 2),
    'mz_limits': (0.005, 2),
    'm4_scan' : [0.01, 0.1, 0.4, 0.8],
    'mz_scan': [0.0075, 0.03, 0.1, 0.4],
    'bp': {
        'm4':0.1,
        'mz':0.03,
        'alpha_dark':0.25,
        'epsilon':np.sqrt(2e-10/alphaQED),
        'Umu4_2':8e-9,
        'Ud4_2':None,
    },
    'ticks': {
        'mz': np.linspace(0.005, 0.095, 10),
        'epsilon': np.geomspace(1e-5, 1e-1, 11),

        'm4': np.linspace(0.0358, 1, 10),
        'Umu4_2': np.geomspace(1e-10, 1e-4, 11),
    }
    # 'lower_bound_Vmu4_alpha_epsilon2': 10**(-21),
    # 'lower_bound_Vmu4': 10**(-10),
    # 'lower_bound_epsilon': 10**(-5),
}


likelihood_calculation_pars = {
'heavy_mz_epsilon': { 
    'name': 'mz_epsilon',
    'hierarchy': 'heavy',
    'analysis_name': 'tpc',
    'pars': {
        'm4':physics_parameters['heavy']['bp']['m4'],
        'mz':physics_parameters['heavy']['ticks']['mz'],
        'alpha_dark':physics_parameters['heavy']['bp']['alpha_dark'],
        'epsilon':physics_parameters['heavy']['ticks']['epsilon'],
        'Umu4_2':physics_parameters['heavy']['bp']['Umu4_2'],
        'Ud4_2':physics_parameters['heavy']['bp']['Ud4_2'],
    },
    'additional_vars': 'cut_based',
},
'heavy_m4_Umu4_2': { 
    'name': 'm4_Umu4_2',
    'hierarchy': 'heavy',
    'analysis_name': 'tpc',
    'pars': {
        'm4':physics_parameters['heavy']['ticks']['m4'],
        'mz':physics_parameters['heavy']['bp']['mz'],
        'alpha_dark':physics_parameters['heavy']['bp']['alpha_dark'],
        'epsilon':physics_parameters['heavy']['bp']['epsilon'],
        'Umu4_2':physics_parameters['heavy']['ticks']['Umu4_2'],
        'Ud4_2':physics_parameters['heavy']['bp']['Ud4_2'],
    },
    'additional_vars': 'cut_based',
},
'light_mz_epsilon': { 
    'name': 'mz_epsilon',
    'hierarchy': 'light',
    'analysis_name': 'nueccqe_fgd',
    'pars': {
        'm4':physics_parameters['light']['bp']['m4'],
        'mz':physics_parameters['light']['ticks']['mz'],
        'alpha_dark':physics_parameters['light']['bp']['alpha_dark'],
        'epsilon':physics_parameters['light']['ticks']['epsilon'],
        'Umu4_2':physics_parameters['light']['bp']['Umu4_2'],
        'Ud4_2':physics_parameters['light']['bp']['Ud4_2'],
    },
    'additional_vars': 'cut_based',
},
'light_m4_Umu4_2': { 
    'name': 'm4_Umu4_2',
    'hierarchy': 'light',
    'analysis_name': 'nueccqe_fgd',
    'pars': {
        'm4':physics_parameters['light']['ticks']['m4'],
        'mz':physics_parameters['light']['bp']['mz'],
        'alpha_dark':physics_parameters['light']['bp']['alpha_dark'],
        'epsilon':physics_parameters['light']['bp']['epsilon'],
        'Umu4_2':physics_parameters['light']['ticks']['Umu4_2'],
        'Ud4_2':physics_parameters['light']['bp']['Ud4_2'],
    },
    'additional_vars': 'cut_based',
}
}
# bp_h1 = {
#     'm4':0.1,
#     'mz':1.25,
#     'Umu4_2':2.2e-7,
#     'Ud4_2':1,
#     'alpha_dark':0.4,
#     'epsilon':2.1e-2,
# }


for physics_params in physics_parameters.values():
    physics_params['upper_bound_epsilon'] = upper_bound_epsilon
    physics_params['upper_bound_Vmu4'] = upper_bound_Umu4_2
    physics_params['Vmu4_alpha_epsilon2'] = physics_params['bp']['alpha_dark'] *\
                                                         physics_params['bp']['Umu4_2'] *\
                                                         alphaQED *\
                                                         physics_params['bp']['epsilon']**2 
    physics_params['upper_bound_Vmu4_alpha_epsilon2'] = physics_params['upper_bound_Vmu4'] *\
                                                        alphaQED *\
                                                        upper_bound_epsilon**2
    physics_params['upper_bound_Valpha4_alpha_epsilon2'] = alphaQED *\
                                                        upper_bound_epsilon**2


    # physics_params['upper_bound_log10_Vmu4_alpha_epsilon2'] = np.log10(physics_params['upper_bound_Vmu4_alpha_epsilon2'])
    # physics_params['lower_bound_log10_Vmu4_alpha_epsilon2'] = np.log10(physics_params['lower_bound_Vmu4_alpha_epsilon2'])
    # physics_params['upper_bound_log10_Valpha4_alpha_epsilon2'] = np.log10(physics_params['upper_bound_Valpha4_alpha_epsilon2'])

tpc_length = 100 #cm
fgd_length = 36.5 #cm
p0d_length = 240 #cm
lead_layer_thickness = 0.45 #cm
n_lead_layers = 14

p0d_dimensions = [210.3, 223.9, p0d_length]

tpc_fiducial_volume_dimensions = [170, 196, 56.3]
tpc_fiducial_volume = tpc_fiducial_volume_dimensions[0]*\
                      tpc_fiducial_volume_dimensions[1]*\
                      tpc_fiducial_volume_dimensions[2]

tpc_fiducial_volume_gap_x = (p0d_dimensions[0] - tpc_fiducial_volume_dimensions[0])/2
tpc_fiducial_volume_gap_y = (p0d_dimensions[1] - tpc_fiducial_volume_dimensions[1])/2
tpc_fiducial_volume_gap_z_begin = 16 #maybe 15.5
tpc_fiducial_volume_gap_z_end = tpc_length - tpc_fiducial_volume_gap_z_begin - tpc_fiducial_volume_dimensions[2]

tpc_fiducial_volume_endpoints = [[tpc_fiducial_volume_gap_x, 
                                  tpc_fiducial_volume_gap_x + tpc_fiducial_volume_dimensions[0]],
                                 [tpc_fiducial_volume_gap_y, 
                                  tpc_fiducial_volume_gap_y + tpc_fiducial_volume_dimensions[1]],
                                 [p0d_dimensions[2] + tpc_fiducial_volume_gap_z_begin,
                                  p0d_dimensions[2] + tpc_fiducial_volume_gap_z_begin + tpc_fiducial_volume_dimensions[2]]]

detector_splitting = {0: [0, 30.5],
                      1: [30.5, 209.6],
                      2: [209.6, 240.0],
                      3: [tpc_fiducial_volume_endpoints[2][0], tpc_fiducial_volume_endpoints[2][1]],
                      4: [tpc_fiducial_volume_endpoints[2][0] + tpc_length + fgd_length, 
                          tpc_fiducial_volume_endpoints[2][1] + tpc_length + fgd_length],
                      5: [tpc_fiducial_volume_endpoints[2][0] + 2*(tpc_length + fgd_length), 
                          tpc_fiducial_volume_endpoints[2][1] + 2*(tpc_length + fgd_length)]}

geometry_material = {
    'hydrogen': [0, 1, 2],
    'oxygen': [1],
    'carbon': [0, 1, 2],
    'copper': [1],
    'zinc': [1],
    'lead': [0, 2],
    'argon': [3, 4, 5],
}

atomic_mass_gev = {
    'hydrogen': 0.9385,
    'carbon': 11.262,
    'oxygen': 15.016,
    'copper': 59.637921,
    'zinc': 61.35913,
    'lead': 194.4572,
    'argon': 37.54 ,
}

cuts_dict = {
    'cut1' : r'cos $\theta_{ee, beam}$ > 0.99',
    'cut2' : r'$t_{exp}$ < 0.03',
    'cut3' : r'cos $\theta_{ee}$ > 0',
    'cut4' : r'$p_{ee}$ > 0.15',
}

default_kde_pars = {
    'distance': 'log',
    'smoothing': [0.1, 0.1], 
    'kernel': 'epa'
}