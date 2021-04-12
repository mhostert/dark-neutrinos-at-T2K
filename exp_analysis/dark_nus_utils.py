import subprocess             
import itertools
from parameters_dict import physics_parameters

def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()
    print(proc_stdout[0].decode("utf-8"))


def produce_samples_without_scanning(case, D_or_M, neval=100000):
    for m4, mz_prime in itertools.product(physics_parameters[case]['m4_scan'], physics_parameters[case]['mz_scan']):
        dark_gen_run = [f'cd ..; python3 dark_gen.py --M4 {m4} --mzprime {mz_prime} '\
                       f'--UMU4 {physics_parameters[case]["Umu4"]} '\
                       f'--alpha_dark {physics_parameters[case]["alpha_dark"]} '\
                       f'--epsilon2 {physics_parameters[case]["epsilon2"]} '\
                       f'--neval {neval} --noplot --hierarchy {case} --D_or_M {D_or_M}']
        
        subprocess_cmd(dark_gen_run)


def produce_scan_sample(case, D_or_M, neval=1000000):
    mu_gen_run = ['cd ..; python3 mu_gen.py ',\
                   f'--mzprime_min {physics_parameters[case]["mz_limits"][0]} '\
                   f'--mzprime_max {physics_parameters[case]["mz_limits"][1]} '\
                   f'--M4_min {physics_parameters[case]["m4_limits"][0]} '\
                   f'--M4_max {physics_parameters[case]["m4_limits"][1]} '\
                   f'--UMU4 {physics_parameters[case]["Umu4"]} '\
                   f'--alpha_dark {physics_parameters[case]["alpha_dark"]} '\
                   f'--epsilon2 {physics_parameters[case]["epsilon2"]} '\
                   f'--neval {neval} --noplot --hierarchy {case} --D_or_M {D_or_M}']

    subprocess_cmd(mu_gen_run)