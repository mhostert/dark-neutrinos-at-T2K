###############################################################################
def epsilon(da, a0, ml, m):
    """
    Calculate the limit on epsilon for a given difference between
    a(observed) - a(SM theory). Based on equation 6 of
    Endo:2012hp. The fine structure contribution is not included as it
    is negligible.

    da: difference in anomalous magnetic moment, a(experiment) - a(SM).
    a0: fine structure constant.
    ml: mass of the lepton.
    m:  mass of the dark photon.
    """
    from scipy import integrate
    from math import pi
    loop = lambda z, ml = ml, m = m: (2*z*(1 - z)**2*ml**2)/(
        z*m**2 + (1 - z)**2*ml**2)
    return (da/(a0/(2*pi)*integrate.quad(loop, 0, 1)[0]))**0.5

###############################################################################
def limit(label, lep, da, m0 = 1e-3, m1 = 1e2, steps = 100):
    """
    Write out the limits on epsilon for a given file name, lepton flavor,
    and difference in anomalous magnetic moment.

    label: name of the limit file to write.
    lep:   label of the lepton.
    da:    difference in anomalous magnetic moment, a(experiment) - a(SM).
    
    Optional parameters control how many points are calculated on a log scale.
    
    m0:    starting mass of the dark photon.
    m1:    ending mass of the dark photon.
    steps: number of steps.
    """
    from darkcast import pars
    from math import log10
    import numpy
    a0   =  1/137.03599904
    frm  = "%11.4e"
    ttl  = "#%%%is %%%is\n" % (len(frm % 1.0) - 1, len(frm % 1.0))
    out = file(label + ".lmt", "w")
    out.write(ttl % ("mass", "lower"))
    for m in numpy.logspace(log10(m0), log10(m1), steps):
        val = (m, + epsilon(da, a0, pars.mfs[lep], m))
        out.write((" ".join([frm]*len(val)) + "\n") % val)
    out.close()

###############################################################################
if __name__ == "__main__":
    # Electron AMM (3 sigma, Cesium, equation 2 of Bodas:2021fsy).
    limit("AMMe_Bodas2021fsy_cs", "e", (8.7 + 3*3.6)*1e-13)

    # Electron AMM (3 sigma, Rubidium, equation 3 of Bodas:2021fsy).
    limit("AMMe_Bodas2021fsy_rb", "e", (4.8 + 3*3.0)*1e-13)

    # Muon AMM limit (5 sigma, equation 1 of Bodas:2021fsy).
    limit("AMMmu_Bodas2021fsy", "mu", (2.79 + 5*0.76)*1e-9)

    # Muon AMM preferred band (2 sigma, equation 1 of Bodas:2021fsy).
    limit("AMMmu_Bodas2021fsy_2l", "mu", (2.79 - 2*0.76)*1e-9)
    limit("AMMmu_Bodas2021fsy_2u", "mu", (2.79 + 2*0.76)*1e-9)
