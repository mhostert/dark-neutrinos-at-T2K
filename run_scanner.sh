#!/bin/sh

python3 mu_gen.py --hierarchy="light_mediator"\
				--mzprime_min=0.002 \
				--mzprime_max=0.400 \
				--M4_min=0.010 \
				--M4_max=0.800 \
				--alpha_dark=0.25 \
				--alpha_epsilon2=2.0e-10 \
				--UMU4=8.0e-9 \
				--neval=10000\
				--nint=100

python3 mu_gen.py --hierarchy="heavy_mediator"\
				--mzprime_min=0.100 \
				--mzprime_max=5.0 \
				--M4_min=0.010 \
				--M4_max=0.800 \
				--alpha_dark=0.4 \
				--alpha_epsilon2=3.4e-6 \
				--UMU4=2.2e-7 \
				--neval=10000\
				--nint=100

