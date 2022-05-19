# DARKCAST is licensed under the GNU GPL version 2 or later.
# Copyright (C) 2021 Philip Ilten, Yotam Soreq, Mike Williams, and Wei Xue.
import darkcast
notes = """
This limit is a projection for SHiP searches using charged final
states from proton bremsstrahlung using a VMD form factor and was
provided by the authors of Ahdida:2020new.
"""
bibtex = """
@article{Ahdida:2020new,
 author        = "Ahdida, C. and others",
 collaboration = "SHiP",
 title         = "{Sensitivity of the SHiP experiment to dark photons decaying
                  to a pair of charged particles}",
 eprint        = "2011.05115",
 archivePrefix = "arXiv",
 primaryClass  = "hep-ex",
 month         = "11",
 year          = "2020"
}
"""
model = darkcast.Model("dark_photon")
production = darkcast.Production("p_brem")
decay = ["e_e", "mu_mu", "pi+_pi-", "pi+_pi-_pi0_pi0", "pi+_pi-_pi0",
         "pi+_pi-_pi+_pi-", "K_K", "K_K_pi", "other"]
bounds = darkcast.Datasets("reach/SHiP_Ahdida2020new_brem_vmd.lmt")
efficiency = darkcast.Efficiency(lratio = 35.0/50.0)
