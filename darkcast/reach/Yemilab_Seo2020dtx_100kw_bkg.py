# DARKCAST is licensed under the GNU GPL version 2 or later.
# Copyright (C) 2021 Philip Ilten, Yotam Soreq, Mike Williams, and Wei Xue.
import darkcast
notes = """
This limit is a projection for the proposed Yemilab search of
Seo:2020dtx using a di-electron final state and a 100 kW beam, with
background included. Note, other limits, including three photon and
absorption are also projected, but not available in Darkcast. These
limits were provided by the authors.

This is a displaced search where the decay volume length over the
shield length is 20/0.5.
"""
bibtex = """
@article{Seo:2020dtx,
 author        = "Seo, Seon-Hee and Kim, Yeongduk",
 title         = "{Dark Photon Search at Yemilab, Korea}",
 eprint        = "2009.11155",
 archivePrefix = "arXiv",
 primaryClass  = "hep-ph",
 month         = "9",
 year          = "2020"
}
"""
model = darkcast.Model("dark_photon")
production = darkcast.Production("e_brem")
decay = ["e_e"]
bounds = darkcast.Datasets("reach/Yemilab_Seo2020dtx_100kw_bkg.lmt")
efficiency = darkcast.Efficiency(lratio = 20.0/0.5)
