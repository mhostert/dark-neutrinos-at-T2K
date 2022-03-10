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

bp_h1 = {
    'm4':0.1,
    'mz':1.25,
    'Umu4_2':2.2e-7,
    'Ud4_2':1,
    'alpha_d':0.4,
    'epsilon':2.1e-2,
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

tpc_systematic_uncertainties = {'FHC': 0.2, 'RHC': 0.2} # percentage

pot_case_flux = {
    'heavy' : {'FHC': 12.34e20, 'RHC': 6.29e20},
    'light' : {'FHC': 11.92e20, 'RHC': 6.29e20}, #nue CCQE
    # 'light' : {'FHC': 5.738e20}, # single photon
}

tpc_length = 100 #cm
fgd_length = 36.5 #cm
p0d_length = 240 #cm
lead_layer_thickness = 0.45 #cm
n_lead_layers = 14

p0d_dimensions = [210.3, 223.9, p0d_length]

tpc_fiducial_volume_dimensions = [170, 196, 56.3]
tpc_fiducial_volume_gap = 16 #maybe 15.5

tpc_fiducial_volume_endpoints = [[(p0d_dimensions[0] - tpc_fiducial_volume_dimensions[0])/2, 
                                  (p0d_dimensions[0] + tpc_fiducial_volume_dimensions[0])/2],
                                 [(p0d_dimensions[1] - tpc_fiducial_volume_dimensions[1])/2, 
                                  (p0d_dimensions[1] + tpc_fiducial_volume_dimensions[1])/2],
                                 [p0d_dimensions[2] + tpc_fiducial_volume_gap,
                                  p0d_dimensions[2] + tpc_fiducial_volume_gap + tpc_fiducial_volume_dimensions[2]]]

detector_splitting = {0: [0, 30.5],
                      1: [30.5, 209.6],
                      2: [209.6, 240.0],
                      3: [tpc_fiducial_volume_endpoints[2][0], tpc_fiducial_volume_endpoints[2][1]],
                      4: [tpc_fiducial_volume_endpoints[2][0] + tpc_length + fgd_length, 
                          tpc_fiducial_volume_endpoints[2][1] + tpc_length + fgd_length],
                      5: [tpc_fiducial_volume_endpoints[2][0] + 2*(tpc_length + fgd_length), 
                          tpc_fiducial_volume_endpoints[2][1] + 2*(tpc_length + fgd_length)]}

mol2natoms = 6.02214e23 # Avogadro's number
ton2grams = 1e6
geometry_material = {
    'hydrogen': [0, 1, 2],
    'oxygen': [1],
    'carbon': [0, 1, 2],
    'copper': [1],
    'zinc': [1],
    'lead': [0, 2],
    'argon': [3, 4, 5],
}

# to be removed
material_dict = {
    0.9385: 'hydrogen',
    11.262: 'carbon',
    15.016: 'oxygen',
    59.637921: 'copper',
    61.35913: 'zinc',
    194.4572: 'lead',
    37.54 : 'argon',
}

# to be removed
gev_mass = dict(zip(material_dict.values(), material_dict.keys()))

molar_mass = {
    'hydrogen': 1.00784,
    'carbon': 12.0107,
    'oxygen': 15.999,
    'copper': 63.546,
    'zinc': 65.38,
    'lead': 207.2,
    'argon': 39.948
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

mass_material = {
    'heavy' : {
        'hydrogen': 3.3*2*molar_mass['hydrogen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen']),
        'oxygen': 3.3*molar_mass['oxygen']/(2*molar_mass['hydrogen'] + molar_mass['oxygen']),
        'carbon': 8.221,
        'copper': 1.315*0.66, # percentage of copper in typical brass
        'zinc': 1.315*0.34, # percentage of zinc in typical brass
        'lead': 3.634,
        'argon': 0.01},
    'light' : {
        'hydrogen': 0,
        'carbon': 0.9195,
        'argon': 0.01}
}

cuts_dict = {
    'cut1' : r'cos $\theta_{ee, beam}$ > 0.99',
    'cut2' : r'$t_{exp}$ < 0.03',
    'cut3' : r'cos $\theta_{ee}$ > 0',
    'cut4' : r'$p_{ee}$ > 0.15',
}

fgd_mass = 0.9195 # ton
# fgd_mass_full = 1848.6 * 1e-9 * 184**2  * 15
fgd_efficiency = 0.1
# fgd_binning = np.linspace(0.00, 0.2, 21) #nue CCQE
fgd_binning = np.linspace(0.00, 0.3, 61) #single photon
fgd_bin_centers = (fgd_binning[1:] + fgd_binning[:-1])/2
fgd_systematic_uncertainties = {'FHC': 0.23, 'RHC': 0.21} # percentage

# selection = 'carbon'
# ratio_fgd_mass_p0d_carbon = fgd_mass / mass_material['carbon']

# upgrade
fgd1_fid_volume = 174.9 * 174.9 * 14.4
fgd1_fid_volume_gap = 5.75
ratio_fgd2_fgd1 = 1
super_fgd_volume = (192 - fgd1_fid_volume_gap) * (192 - fgd1_fid_volume_gap) * 56

# total_pot_fgd_analysis = pot_case_flux['light']['FHC'] + pot_case_flux['light']['RHC'] #nue CCQE
total_pot_fgd_analysis = pot_case_flux['light']['FHC'] #single photon
pot_before_upgrade = 4e21
pot_after_upgrade = 16e21
total_pot = pot_before_upgrade + pot_after_upgrade

fgd_sensitivity_scale_factor = (1 + ratio_fgd2_fgd1) * total_pot/total_pot_fgd_analysis +\
                               super_fgd_volume * pot_after_upgrade / (fgd1_fid_volume * total_pot_fgd_analysis)