# DARKCAST is licensed under the GNU GPL version 2 or later.
# Copyright (C) 2021 Philip Ilten, Yotam Soreq, Mike Williams, and Wei Xue.
import darkcast
notes = """
This limit was extracted from figure 5 (black curve) of
TheBABAR:2016rlg.

This search is prompt, and is not sensitive to X bosons with lifetimes
large enough to qualify as non-prompt; the efficiency ratio is assumed
to be unity, e.g. t0 = 0 and t1 = infinity.
"""
bibtex = """
@article{TheBABAR:2016rlg,
 author = "Lees, J. P. and others",
 collaboration = "BaBar",
 title = "{Search for a muonic dark force at BABAR}",
 eprint = "1606.03501",
 archivePrefix = "arXiv",
 primaryClass = "hep-ex",
 reportNumber = "BABAR-PUB-16-003, SLAC-PUB-16549",
 doi = "10.1103/PhysRevD.94.011102",
 journal = "Phys. Rev. D",
 volume = "94",
 number = "1",
 pages = "011102",
 year = "2016"
}
"""
model = darkcast.Model("dark_photon")
production = darkcast.Production("e_e")
decay = ["mu_mu"]
bounds = darkcast.Dataset("limits/BaBar_TheBABAR2016rlg.lmt")
efficiency = darkcast.Efficiency(t0 = 0, t1 = float("inf"))
