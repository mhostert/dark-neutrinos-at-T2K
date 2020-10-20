# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cProfile

cProfile.run('exec(open("dark_gen.py").read())','stats')
import pstats
p = pstats.Stats('stats')
p.strip_dirs().sort_stats('cumtime').print_stats(300)