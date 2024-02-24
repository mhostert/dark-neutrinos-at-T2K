import numpy as np
from const import alphaQED

likelihood_levels_1d = {0.68: 1/2,
          0.9: 2.71/2,
          0.95: 3.84/2}

likelihood_levels_2d = {0.68: 2.3/2,
          0.9: 4.61/2,
          0.95: 5.99/2}

atomic_mass_gev = {
    'hydrogen': 0.9385,
    'carbon': 11.262,
    'oxygen': 15.016,
    'copper': 59.637921,
    'zinc': 61.35913,
    'lead': 194.4572,
    'argon': 37.54 ,
}

molar_mass = {
    'hydrogen': 1.00784,
    'carbon': 12.0107,
    'oxygen': 15.999,
    'copper': 63.546,
    'zinc': 65.38,
    'lead': 207.2,
    'argon': 39.948
}

upper_bound_epsilon = 0.05 # model independent constraint on epsilon - probably we can push to 3.5%
upper_bound_Umu4_2 = 0.03 # 

physics_parameters = {}

physics_parameters['heavy'] = {
    'm4_limits': (0.005, 1),
    'mz_limits': (0.005, 10),
    'm4_scan' : [0.01, 0.1, 0.5],
    'mz_scan': [0.02, 0.2, 1, 5],
    'bp': {
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
    },
    'lower_bound_Vmu4_alpha_epsilon2': 10**(-17),
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
    },
    'lower_bound_Vmu4_alpha_epsilon2': 10**(-21),
    'lower_bound_Vmu4': 10**(-10),
    'lower_bound_epsilon': 10**(-5),
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

#####
# Detector numbers

p0d_dimensions = [210.3, 223.9, 240]

tpc_outer_volume = [210.3, 223.9, 100]
tpc_active_volume = [186, 206, 78.1]
tpc_fiducial_volume = [170, 196, 56.3]

fgd_outer_volume = [230, 240, 36.5]
fgd_active_volume = [186.4, 186.4, 30.3]
fgd_fiducial_volume = [174.9, 174.9, 28.76]

tpc_total_fiducial_volume = tpc_fiducial_volume[0]*tpc_fiducial_volume[1]*tpc_fiducial_volume[2]
tpc_total_active_volume = tpc_active_volume[0]*tpc_active_volume[1]*tpc_active_volume[2]

fgd_total_fiducial_volume = fgd_fiducial_volume[0]*fgd_fiducial_volume[1]*fgd_fiducial_volume[2]
fgd_total_active_volume = fgd_active_volume[0]*fgd_active_volume[1]*fgd_active_volume[2]
fgd_fiducial_volume_factor = fgd_total_fiducial_volume/fgd_total_active_volume

detector_splitting_x = {0: [-p0d_dimensions[0]/2, p0d_dimensions[0]/2],
                      1: [-p0d_dimensions[0]/2, p0d_dimensions[0]/2],
                      2: [-p0d_dimensions[0]/2, p0d_dimensions[0]/2],
                      3: [-tpc_active_volume[0]/2, tpc_active_volume[0]/2],
                      4: [-fgd_active_volume[0]/2, fgd_active_volume[0]/2],
                      5: [-tpc_active_volume[0]/2, tpc_active_volume[0]/2],
                      6: [-fgd_active_volume[0]/2, fgd_active_volume[0]/2],
                      7: [-tpc_active_volume[0]/2, tpc_active_volume[0]/2],}

detector_splitting_y = {0: [-p0d_dimensions[1]/2, p0d_dimensions[1]/2],
                      1: [-p0d_dimensions[1]/2, p0d_dimensions[1]/2],
                      2: [-p0d_dimensions[1]/2, p0d_dimensions[1]/2],
                      3: [-tpc_active_volume[1]/2, tpc_active_volume[1]/2],
                      4: [-fgd_active_volume[1]/2, fgd_active_volume[1]/2],
                      5: [-tpc_active_volume[1]/2, tpc_active_volume[1]/2],
                      6: [-fgd_active_volume[1]/2, fgd_active_volume[1]/2],
                      7: [-tpc_active_volume[1]/2, tpc_active_volume[1]/2],}

detector_splitting_z = {0: [0, 30.5],
                       1: [30.5, 209.6],
                       2: [209.6, p0d_dimensions[2]],
                       3: [p0d_dimensions[2] + (tpc_outer_volume[2] - tpc_active_volume[2])/2, 
                           p0d_dimensions[2] + (tpc_outer_volume[2] + tpc_active_volume[2])/2],
                       4: [p0d_dimensions[2] + tpc_outer_volume[2] + (fgd_outer_volume[2] - fgd_active_volume[2])/2, 
                           p0d_dimensions[2] + tpc_outer_volume[2] + (fgd_outer_volume[2] + fgd_active_volume[2])/2],
                       5: [p0d_dimensions[2] + tpc_outer_volume[2] + fgd_outer_volume[2] + (tpc_outer_volume[2] - tpc_active_volume[2])/2, 
                           p0d_dimensions[2] + tpc_outer_volume[2] + fgd_outer_volume[2] + (tpc_outer_volume[2] + tpc_active_volume[2])/2],
                       6: [p0d_dimensions[2] + 2*tpc_outer_volume[2] + fgd_outer_volume[2] + (fgd_outer_volume[2] - fgd_active_volume[2])/2, 
                           p0d_dimensions[2] + 2*tpc_outer_volume[2] + fgd_outer_volume[2] + (fgd_outer_volume[2] + fgd_active_volume[2])/2],
                       7: [p0d_dimensions[2] + 2*(tpc_outer_volume[2] + fgd_outer_volume[2]) + (tpc_outer_volume[2] - tpc_active_volume[2])/2, 
                           p0d_dimensions[2] + 2*(tpc_outer_volume[2] + fgd_outer_volume[2]) + (tpc_outer_volume[2] + tpc_active_volume[2])/2]}

tpc_fiducial_volume_gap_z_begin = 16 #maybe 15.5
tpc_fiducial_volume_endpoints = [[-tpc_fiducial_volume[0]/2, tpc_fiducial_volume[0]/2],
                                 [-tpc_fiducial_volume[1]/2, tpc_fiducial_volume[1]/2],
                                 [p0d_dimensions[2] + tpc_fiducial_volume_gap_z_begin,
                                  p0d_dimensions[2] + tpc_fiducial_volume_gap_z_begin + tpc_fiducial_volume[2]]]

geometry_material = {
    'hydrogen': [0, 1, 2, 4, 6],
    'oxygen': [1, 4, 6],
    'carbon': [0, 1, 2, 4, 6],
    'copper': [1],
    'zinc': [1],
    'lead': [0, 2],
    'argon': [3, 5, 7],
}

n_fgd1_xy_modules = 15
n_fgd2_xy_modules = 7
n_fgd2_water_modules = 6

fgd_xy_module_carbon = 1.8486*fgd_active_volume[0]*fgd_active_volume[1]*1e-6 #ton, the first is the density in mg/cm^2
fgd_xy_module_hydrogen = 0.1579*fgd_active_volume[0]*fgd_active_volume[1]*1e-6
fgd_xy_module_oxygen = 0.0794*fgd_active_volume[0]*fgd_active_volume[1]*1e-6

fgd_water_module_carbon = 0.422*fgd_active_volume[0]*fgd_active_volume[1]*1e-6
fgd_water_module_hydrogen = 0.2916*fgd_active_volume[0]*fgd_active_volume[1]*1e-6
fgd_water_module_oxygen = 2.0601*fgd_active_volume[0]*fgd_active_volume[1]*1e-6

fgd1_carbon_mass = fgd_xy_module_carbon*n_fgd1_xy_modules
fgd2_carbon_mass = fgd_xy_module_carbon*n_fgd2_xy_modules+fgd_water_module_carbon*n_fgd2_water_modules

fgd1_hydrogen_mass = fgd_xy_module_hydrogen*n_fgd1_xy_modules
fgd2_hydrogen_mass = fgd_xy_module_hydrogen*n_fgd2_xy_modules+fgd_water_module_hydrogen*n_fgd2_water_modules

fgd1_oxygen_mass = fgd_xy_module_oxygen*n_fgd1_xy_modules
fgd2_oxygen_mass = fgd_xy_module_oxygen*n_fgd2_xy_modules+fgd_water_module_oxygen*n_fgd2_water_modules

p0d_hydrogen_mass = 3.3*2*molar_mass['hydrogen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen'])
p0d_oxygen_mass = 3.3*molar_mass['oxygen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen'])
p0d_carbon_mass = 8.221

p0d_p0dule_mass = 0.192
p0d_brass_sheet_mass = 0.1
p0d_water_bag_mass = 0.1
p0d_lead_sheet_mass = 0.222

n_p0dule_up_ecal = 7
n_lead_up_ecal = 7

n_p0dule_central = 26
n_brass_central = 25
n_water_central = 25

n_p0dule_down_ecal = 7
n_lead_down_ecal = 7

hydrogen_in_p0dule = 2/14
carbon_in_p0dule = 12/14
copper_in_brass = 0.66
zinc_in_brass = 1 - copper_in_brass
hydrogen_in_water = 2/18
oxygen_in_water = 16/18

active_argon_three_tpcs = 0.016020

tpc_masses = {
    # 'hydrogen': (n_p0dule_up_ecal+n_p0dule_central+n_p0dule_down_ecal)*p0d_p0dule_mass*hydrogen_in_p0dule +
    #             n_water_central*p0d_water_bag_mass*hydrogen_in_water +
    #             fgd1_hydrogen_mass + fgd2_hydrogen_mass,
    'hydrogen': 0,
    'oxygen': n_water_central*p0d_water_bag_mass*oxygen_in_water + 
              fgd1_oxygen_mass + fgd2_oxygen_mass,
    'carbon': (n_p0dule_up_ecal+n_p0dule_central+n_p0dule_down_ecal)*p0d_p0dule_mass*carbon_in_p0dule + 
              fgd1_carbon_mass + fgd2_carbon_mass,
    'copper': p0d_brass_sheet_mass*n_brass_central*copper_in_brass,
    'zinc': p0d_brass_sheet_mass*n_brass_central*zinc_in_brass,
    'lead': p0d_lead_sheet_mass * (n_lead_up_ecal + n_lead_down_ecal),
    'argon': active_argon_three_tpcs,
}

mass_material = {
    'hydrogen': [n_p0dule_up_ecal*p0d_p0dule_mass*hydrogen_in_p0dule, 
                 n_p0dule_central*p0d_p0dule_mass*hydrogen_in_p0dule + n_water_central*p0d_water_bag_mass*hydrogen_in_water, 
                 n_p0dule_down_ecal*p0d_p0dule_mass*hydrogen_in_p0dule, 
                 fgd1_hydrogen_mass,
                 fgd2_hydrogen_mass],
    'oxygen': [n_water_central*p0d_water_bag_mass*oxygen_in_water, 
               fgd1_oxygen_mass, 
               fgd2_oxygen_mass],
    'carbon': [n_p0dule_up_ecal*p0d_p0dule_mass*carbon_in_p0dule,
               n_p0dule_central*p0d_p0dule_mass*carbon_in_p0dule,
               n_p0dule_down_ecal*p0d_p0dule_mass*carbon_in_p0dule,
               fgd1_carbon_mass,
               fgd2_carbon_mass],
    'copper': [p0d_brass_sheet_mass*n_brass_central*copper_in_brass],
    'zinc': [p0d_brass_sheet_mass*n_brass_central*zinc_in_brass],
    'lead': [p0d_lead_sheet_mass*n_lead_up_ecal, p0d_lead_sheet_mass*n_lead_down_ecal],
    'argon': [active_argon_three_tpcs/3, active_argon_three_tpcs/3, active_argon_three_tpcs/3],
}
mass_weights = {key:np.array(value)/np.array(value).sum() for key, value in mass_material.items()}

tpc_masses_light_case = {'argon': active_argon_three_tpcs*tpc_total_fiducial_volume/tpc_total_active_volume}

new_tpc_fid_volume = 2 *\
                 (200 - 2*(tpc_active_volume[0]-tpc_fiducial_volume[0])) *\
                 (80 - 2*(tpc_active_volume[1]-tpc_fiducial_volume[1])) *\
                 (180 - 2*(tpc_active_volume[2]-tpc_fiducial_volume[2]))

tpc_masses_projection_phase2 = {
    'argon': tpc_masses_light_case['argon'] * (1 + new_tpc_fid_volume/(3*tpc_total_fiducial_volume))
}

# fgd masses
fgd1_masses = {
    # 'hydrogen': fgd1_hydrogen_mass *fgd_fiducial_volume_factor,
              'hydrogen': 0,
              'carbon': fgd1_carbon_mass *fgd_fiducial_volume_factor,
              'oxygen': fgd1_oxygen_mass *fgd_fiducial_volume_factor,
              'argon': 0
             }

fgd_masses_projection_phase1 = {
    # 'hydrogen': (fgd1_hydrogen_mass + fgd2_hydrogen_mass)*fgd_fiducial_volume_factor,
                                'hydrogen': 0,
                                'carbon': (fgd1_carbon_mass + fgd2_carbon_mass)*fgd_fiducial_volume_factor,
                                'oxygen': (fgd1_oxygen_mass + fgd2_oxygen_mass)*fgd_fiducial_volume_factor,
                                'argon': 0
                                }

super_fgd_fiducial_volume = (192 - 2*(fgd_active_volume[0]-fgd_fiducial_volume[0])) *\
                            (56 - 2*(fgd_active_volume[1]-fgd_fiducial_volume[1])) *\
                            (192 - 2*(fgd_active_volume[2]-fgd_fiducial_volume[2]))

fgd_projection_volume_scale_factor_phase2 = super_fgd_fiducial_volume/fgd_total_fiducial_volume

fgd_masses_projection_phase2 = {
                               # 'hydrogen': (fgd1_hydrogen_mass + fgd2_hydrogen_mass)*fgd_fiducial_volume_factor + fgd1_hydrogen_mass*fgd_projection_volume_scale_factor_phase2,
                                'hydrogen': 0,
                                'carbon': (fgd1_carbon_mass + fgd2_carbon_mass)*fgd_fiducial_volume_factor + fgd1_carbon_mass*fgd_projection_volume_scale_factor_phase2,
                                'argon': 0
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