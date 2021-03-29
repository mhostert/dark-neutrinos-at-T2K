#!/bin/sh

python3 mu_gen.py --hierarchy="light_mediator"\
				--mzprime_min=0.0299 \
				--mzprime_max=0.0301 \
				--M4_min=0.05 \
				--M4_max=0.5 \
				--alpha_dark=0.4 \
				--epsilon2=4.6e-4 \
				--UMU4=2.2e-7 \
				--neval=100000\
				--nint=100


# python3 mu_gen.py --hierarchy="heavy_mediator"\
# 				--mzprime_min=0.0299 \
# 				--mzprime_max=0.0301 \
# 				--M4_min=0.005 \
# 				--M4_max=0.5 \
# 				--alpha_dark=0.4 \
# 				--epsilon2=4.6e-4 \
# 				--UMU4=2.2e-7 \
# 				--neval=100000\
# 				--nint=100


