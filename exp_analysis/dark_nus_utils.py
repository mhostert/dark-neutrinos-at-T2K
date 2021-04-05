import itertools
from parameters_dict import physics_parameters

def produce_samples_without_scanning(case, neval=100000):
    for m4, mz_prime in itertools.product(physics_parameters[case]['m4_scan'], physics_parameters[case]['mz_scan']):
        print(m4, mz_prime)
        dark_gen_run = f'cd ..; python dark_gen.py --M4 {m4} --mzprime {mz_prime} '\
                       f'--UMU4 {physics_parameters[case]["Umu4"]} '\
                       f'--alpha_dark {physics_parameters[case]["alpha_dark"]} '\
                       f'--epsilon2 {physics_parameters[case]["epsilon2"]} '\
                       f'--neval {neval} --noplot --hierarchy {case}_mediator'
        stream = os.popen(dark_gen_run)
        print(stream.read())
        
def produce_scan_sample(case, neval=1000000):
    mu_gen_run = 'cd ..; python mu_gen.py '\
                   f'--mzprime_min {physics_parameters[case]["mz_limits"][0]} '\
                   f'--mzprime_max {physics_parameters[case]["mz_limits"][1]} '\
                   f'--M4_min {physics_parameters[case]["m4_limits"][0]} '\
                   f'--M4_max {physics_parameters[case]["m4_limits"][1]} '\
                   f'--UMU4 {physics_parameters[case]["Umu4"]} '\
                   f'--alpha_dark {physics_parameters[case]["alpha_dark"]} '\
                   f'--epsilon2 {physics_parameters[case]["epsilon2"]} '\
                   f'--neval {neval} --noplot --hierarchy {case}_mediator'
    stream = os.popen(mu_gen_run)
    print(stream.read())