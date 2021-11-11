This data release contains one ROOT file with two 2D histograms showing the electron/positron smearing matrices for momentum and angle, obtained using the selection from the ND280 nue CC inclusive analysis [*]. The histograms are also provided as png files for completeness.

The Monte Carlo events that are used to produce these matrices are generated using NEUT 5.3.2, with a focus on T2K neutrino spectrum energies (Enu peaking at 600 MeV). The selected neutrino interactions occur in FGD1 fiducial volume and the momentum of the corresponding electron is reconstructed using the TPC2. See details in the paper [*].

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! See WARNINGS at the end of this file !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

=======================
DESCRIPTION
=======================

**Momentum** 

Distribution of the true vs reconstructed values of momentum for all selected electrons/positrons (true and reconstructed) from $\nu_e$CC, $\bar\nu_e$CC and gamma selections in FHC and RHC beam modes. The true (reconstructed) momentum is ranging from 0 to 2000 (2500) MeV/c with a bin width of 50 MeV/c.

**Angle**

Distribution of the true vs reconstructed values of electron/positron angle for all selected electrons/positrons (true and reconstructed) from $\nu_e$CC, $\bar\nu_e$CC and gamma selections in FHC and RHC beam modes. The angle is defined with respect to the incoming neutrino direction (true: truth direction; reconstructed: defined as the line between mean neutrino production point in the decay tunnel and reconstructed interaction vertex). The angle is ranging from 0 to 1.6 rad with a bin width of 16 mrad.


=======================
HOW TO USE
=======================

For smearing purpose, the absolute scale of the histogram has no particular meaning. For a given true electron momentum (angle), the corresponding bin slice can be used to generate its associated reconstructed momentum (angle).

For each bin i, the associated MC statistical uncertainty can be recovered with the ROOT function GetBinError(i). It may be useful to properly estimate how precise is the estimation of the smearing for a given true momentum slice, based on the available MC statistic. Rebinning may be necessary in case where the slice has not enough statistic and it is up to the analyser(s) to perform this eventual rebinning based on their use case.


**Sample code**

Let's consider p_true = 540 MeV/c and we want to generate associated p_reco. One can get the corresponding bin slice histogram with the following code in ROOT prompt or macro:

   TFile f("smearing.root");
   TH2F* hmom = (TH2F*)f.Get("Momentum");
   // Find the correct true momentum bin
   int bin540 = hmom->GetXaxis()->FindBin(540);
   // Project the 2D histogram for this particular bin
   TH1D* hreco_truemom540 = hmom->ProjectionY("hreco_truemom540", bin540, bin540);

The histogram can then be used to generate p_reco:

   double preco = hreco_truemom540->GetRandom()


=======================
WARNINGS AND CAVEATS
=======================

The associated selection efficiency are not illustrated in the smearing matrices, but may be coarsely estimated from Figures 15 and 16 in the T2K nue CC inclusive cross-section paper [*]. 

The correlations between electron momentum and direction (e.g. the fact that momentum resolution may vary depending on the track direction), as well as the impact of the presence of additional tracks, are not considered. The related systematic uncertainties are completely disregarded. 

These effects may have a non-negligible impact on the comparison to reconstructed T2K event distributions. In contrast to T2K cross-section measurements and other official T2K analyses, the information provided here is therefore only suitable for providing a qualitative comparison with the reconstructed distributions and a quantitative comparison should not be attempted. 

We strongly encourage any users of this data release to present these caveats alongside any public comparison to T2K data. 



[*]: J. High Energ. Phys. 2020, 114 (2020), arXiv:2002.11986
