"""
This module contains functions for calculating photometric effective
temperatures.

Contents:
    | ``given_dr2_BpmRp_AV_get_Teff_Curtis2020``
"""
import os
import numpy as np, pandas as pd
from gyrointerp.paths import DATADIR

def given_dr2_BpmRp_AV_get_Teff_Curtis2020(dr2_BpmRp, A_V):
    """
    Empirical color-temperature relation from Appendix A of Curtis+2020.
    Visible in their Figure 11.  Coefficients from their Table 4.

    This relation was constructed using benchmark stars from Brewer+2016a,
    Boyajian+2012, and Mann+2015 (which should also be cited).

    It is calibrated in the range 0.55 < (BP-RP)0 < 3.25, and has a scatter of
    about +/- 50 K.

    It's important that the passed BP-RP colors are from Gaia DR2. For FGK
    stars in a few test clusters (eg. NGC-3532), the typical offset is +0.02
    mag and color dependent.  For late K dwarf and M-dwarfs, it flips, and gets
    down to -0.1 mag at BP-RP of >2 (SpType>M1V).  This translates to a >100 K
    systematic error if you use the wrong Gaia data release.

    See https://www.cosmos.esa.int/web/gaia/edr3-passbands for a description of
    why exactly the Gaia passbands changed between reductions.

    Args:
        dr2_BpmRp (np.ndarray): *observed* Gaia DR2 BP-RP array.

        A_V (float): mean reddening.

    Returns:

        Teff (np.ndarray): array of effective temperatures.
    """
    c0 = -416.585
    c1 = 39780.0
    c2 = -84190.5
    c3 = 85203.9
    c4 = -48225.9
    c5 = 15598.5
    c6 = -2694.76
    c7 = 192.865

    # Figure 7 caption from Curtis+2020.  This is way simpler than using the
    # Gaia Collaboration's coefficients.
    E_BPmRP = 0.415 * A_V

    dr2_BpmRp0 = dr2_BpmRp - E_BPmRP

    Teff = (
        c0*(dr2_BpmRp0)**0 +
        c1*(dr2_BpmRp0)**1 +
        c2*(dr2_BpmRp0)**2 +
        c3*(dr2_BpmRp0)**3 +
        c4*(dr2_BpmRp0)**4 +
        c5*(dr2_BpmRp0)**5 +
        c6*(dr2_BpmRp0)**6 +
        c7*(dr2_BpmRp0)**7
    )

    bad = (
        (dr2_BpmRp0 < 0.55)
        |
        (dr2_BpmRp0 > 3.25)
    )
    Teff[bad] = np.nan

    return Teff


def _given_VmKs_get_Teff(VmKs):
    """
    Interpolate effective temperatures from the Mamajek table and (V-Ks)0.
    (less reddening dependence than B-V).
    """
    mamajekpath = os.path.join(DATADIR, "literature",
                               "EEM_dwarf_UBVIJHK_colors_Teff_20220416.txt")
    mamajek_df = pd.read_csv(
        mamajekpath, comment='#', delim_whitespace=True
    )
    sel = (
        (mamajek_df['V-Ks'] != ".....")
        &
        (mamajek_df['V-Ks'] != "....")
        &
        (mamajek_df['V-Ks'] != "...")
        &
        (mamajek_df["Teff"] > 3300)
        &
        (mamajek_df["Teff"] < 7400)
    )
    mamajek_df = mamajek_df[sel]
    mamajek_df = mamajek_df.reset_index(drop=True)

    poly_model = np.polyfit(
        mamajek_df['V-Ks'].astype(float), mamajek_df['Teff'].astype(float), 7
    )
    testmodel = np.polyval(poly_model, mamajek_df['V-Ks'].astype(float))

    Teff = np.polyval(poly_model, VmKs)

    return Teff


def _given_GmKs_get_Teff(GmKs):
    """
    Interpolate effective temperatures from the Mamajek table and (G-Ks)0.
    """
    mamajekpath = os.path.join(DATADIR, "literature",
                               "EEM_dwarf_UBVIJHK_colors_Teff_20220416.txt")
    mamajek_df = pd.read_csv(
        mamajekpath, comment='#', delim_whitespace=True
    )
    sel = (
        (mamajek_df['G-V'] != ".....")
        &
        (mamajek_df['G-V'] != "....")
        &
        (mamajek_df['G-V'] != "...")
        &
        (mamajek_df['V-Ks'] != ".....")
        &
        (mamajek_df['V-Ks'] != "....")
        &
        (mamajek_df['V-Ks'] != "...")
        &
        (mamajek_df["Teff"] > 3300)
        &
        (mamajek_df["Teff"] < 7400)
    )
    mamajek_df = mamajek_df[sel]
    mamajek_df = mamajek_df.reset_index(drop=True)

    mamajek_df['G-Ks'] = mamajek_df['G-V'].astype(float) + mamajek_df['V-Ks'].astype(float)

    poly_model = np.polyfit(
        mamajek_df['G-Ks'].astype(float), mamajek_df['Teff'].astype(float), 7
    )
    testmodel = np.polyval(poly_model, mamajek_df['G-Ks'].astype(float))

    Teff = np.polyval(poly_model, GmKs)

    return Teff
