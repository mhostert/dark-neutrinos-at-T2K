# DARKCAST is licensed under the GNU GPL version 2 or later.
# Copyright (C) 2021 Philip Ilten, Yotam Soreq, Mike Williams, and Wei Xue.
import darkcast
notes = """
This is the preferred lower band for the dark photon, not a limit,
given the difference between the measured muon anomalous magnetic
moment and SM prediction. This band was calculated using AMM.py and
the values of equation 1 from Bodas:2021fsy at 2 sigma around the
nominal value.

Given the nature of the limit, the efficiency ratio is unity, e.g. t0
= 0 and t1 = infinity.
"""
bibtex = """
@article{Bodas:2021fsy,
 author = "Bodas, Arushi and Coy, Rupert and King, Simon J. D.",
 title = "{Solving the electron and muon $g-2$ anomalies in $Z'$ models}",
 eprint = "2102.07781",
 archivePrefix = "arXiv",
 primaryClass = "hep-ph",
 reportNumber = "ULB-TH/21-01",
 month = "2",
 year = "2021"
}
"""
model = darkcast.Model("dark_photon")
production = darkcast.Production("mu_mu")
decay = "none"
bounds = darkcast.Dataset("limits/AMMmu_Bodas2021fsy_2l.lmt")
efficiency = darkcast.Efficiency(t0 = 0, t1 = float("inf"))
