import os
import subprocess             
import itertools
import pickle
from time import time, process_time

from parameters_dict import physics_parameters
from exp_analysis_class import exp_analysis


# run shell commands from notebook
def subprocess_cmd(command, verbose=2):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout,stderr = process.communicate()
    if verbose==2:  
        # print(command)
        print(stdout.decode("utf-8"))
        print(stderr.decode("utf-8"))
    elif verbose==1:
        if len(stderr.decode("utf-8"))>2:
            # print(command)
            print('n',stderr.decode("utf-8"),'m')

def produce_samples_without_scanning(case, D_or_M, neval=100000, exp='nd280_nu'):
    for m4, mz_prime in itertools.product(physics_parameters[case]['m4_scan'], physics_parameters[case]['mz_scan']):
        print(f"Generating events for m4={m4} GeV and mzprime={mz_prime}")
        dark_gen_run = [f'cd ..; python3 dark_gen.py --M4 {m4} --mzprime {mz_prime} '\
                       f'--UMU4 {physics_parameters[case]["bp"]["Umu4_2"]} '\
                       f'--alpha_dark {physics_parameters[case]["bp"]["alpha_dark"]} '\
                       f'--epsilon2 {physics_parameters[case]["bp"]["epsilon"]**2} '\
                       f'--neval {neval} --noplot --hierarchy {case} --D_or_M {D_or_M} '\
                       f'--exp {exp}']
        subprocess_cmd(dark_gen_run)


def produce_samples_without_scanning_m4mz(case, D_or_M, m4mz=[0.15, 0.03], neval=100000, exp='nd280_nu'):
    print(f"Generating events for m4={m4mz[0]} GeV and mzprime={m4mz[1]}")
    dark_gen_run = [f'cd ..; python3 dark_gen.py --M4 {m4mz[0]} --mzprime {m4mz[1]} '\
                   f'--UMU4 {physics_parameters[case]["bp"]["Umu4_2"]} '\
                   f'--alpha_dark {physics_parameters[case]["bp"]["alpha_dark"]} '\
                   f'--epsilon2 {physics_parameters[case]["bp"]["epsilon"]**2} '\
                   f'--neval {neval} --noplot --hierarchy {case} --D_or_M {D_or_M} '\
                   f'--exp {exp}']
    subprocess_cmd(dark_gen_run)


def produce_scan_sample(case, D_or_M, neval=1000000, exp='nd280_nu'):
    mu_gen_run = ['cd ..; python3 mu_gen.py '\
                   f'--mzprime_min {physics_parameters[case]["mz_limits"][0]} '\
                   f'--mzprime_max {physics_parameters[case]["mz_limits"][1]} '\
                   f'--M4_min {physics_parameters[case]["m4_limits"][0]} '\
                   f'--M4_max {physics_parameters[case]["m4_limits"][1]} '\
                   f'--UMU4 {physics_parameters[case]["bp"]["Umu4_2"]} '\
                   f'--alpha_dark {physics_parameters[case]["bp"]["alpha_dark"]} '\
                   f'--epsilon2 {physics_parameters[case]["bp"]["epsilon"]**2} '\
                   f'--neval {neval} --noplot --hierarchy {case} --D_or_M {D_or_M} '\
                   f'--exp {exp}']

    subprocess_cmd(mu_gen_run)
                  
def load_datasets(hierarchies=['heavy', 'light'], D_or_Ms=['dirac', 'majorana'], fluxes=['FHC', 'RHC'], dump=False, timeit=False, direct_load_objects=False, build_ball_tree=False, distance='log', load_grid=True):
    assert not (dump and direct_load_objects)
    if type(hierarchies) is not list:
        hierarchies = [hierarchies]
    if type(D_or_Ms) is not list:
        D_or_Ms = [D_or_Ms]
    if type(fluxes) is not list:
        fluxes = [fluxes]
    
    my_exp_analyses = {}
    for hierarchy, D_or_M, flux in itertools.product(hierarchies, D_or_Ms, fluxes):
        print(hierarchy, D_or_M, flux)
        if hierarchy == 'light':
            smearing = True
        else:
            smearing = False
        if timeit:
            start = time()
            start_process = process_time()
        this_exp_analyis = exp_analysis(hierarchy, D_or_M, flux)
        if dump or direct_load_objects:
            filename_pickle = f'{this_exp_analyis.base_folder}exp_analysis_objects/{this_exp_analyis.hierarchy}_{this_exp_analyis.D_or_M}.pckl'
        if not direct_load_objects:
            this_exp_analyis.load_df_base(1000000,
                                          build_ball_tree=build_ball_tree,
                                          distance=distance,
                                          smearing=smearing)
            if load_grid:
                this_exp_analyis.load_grid_dfs()
            my_exp_analyses[f'{hierarchy}_{D_or_M}_{flux}'] = this_exp_analyis
            if dump:
                os.makedirs(f'{this_exp_analyis.base_folder}exp_analysis_objects/', exist_ok=True)
                f = open(filename_pickle, 'wb+')
                pickle.dump(this_exp_analyis, f)
                f.close()
        else:
            f = open(filename_pickle, 'rb')
            my_exp_analyses[f'{hierarchy}_{D_or_M}_{flux}'] = pickle.load(f)
            f.close()
        if timeit:
            end = time()
            end_process = process_time()
            print(f'Wall time: {end - start} s, CPU time: {end_process - start_process}')  
    return my_exp_analyses
