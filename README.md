    ###############################################################
    #    ______           _        _   _                          # 
    #    |  _  \         | |      | \ | |                         #
    #    | | | |__ _ _ __| | __   |  \| | _____      _____        #
    #    | | | / _  | ___| |/ /   | .   |/ _ \ \ /\ / / __|       #
    #    | |/ / (_| | |  |   <    | |\  |  __/\ V  V /\__ \       #
    #    |___/ \__,_|_|  |_|\_\   \_| \_/\___| \_/\_/ |___/       #
    #                                                             #
    ###############################################################

Event generator for dark neutrino events

# SETUP

Requires VEGAS to be working in python with "import vegas". 

# USAGE

To generate "Nevents" HEPevt events with a 4-th neutrino mass of "m_4", Zprime mass of "m_zprime" and proper decay length for the 4-th neutrino of "l_p", you can simply run

./dark_gen.py --M4 0.42 --mzprime 0.03 --nevents 100 --noplot --ldecay 0.05 

This will create a file "data/uboone/3plus1/m4_0.42_mzprime_0.03/MC_m4_0.42_mzprime_0.03.dat" for m_zprime=0.03 GeV, m_4 = 0.420 GeV, l_dec=0.05 meters, and will contain 100 events in HEPevt format.

######
## Options

-h: prints out README.md

--nodif: use only coherent events

--noplot: do not plot distributions in "plots/EXP/MODEL/POINT/"

######
## Options with arguments (--a=default) 

--mzprime=1.215 mass of the zprime in GeV

--mlight=0.140 mass of the light sterile in GeV

--mheavy=1000000 mass of the heavy sterile in GeV (if not set, 3+1 model is used instead)

--ldecay=0.05 proper decay length of the decaying sterile (nu5 in 3+2 and nu4 in 3+1)

--nevents=50 number of events to print to file in a HEPevt format

--exp=uboone experiment to be used (supports "miniboone" and "uboone")
# nicgen
