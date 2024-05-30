"""
These functions return the rotation periods and effective temperatures for
single members of benchmark open clusters.  These are the functions used to
acquire and clean the data behind Figure 1 of BPH23.  Generally, most users
will want to just download the data from the paper, rather than using these
functions.  An additional dependency if you do want to use them is the
``cdips`` module, best installed via a setup.py install from
https://github.com/lgbouma/cdips.

Get all available cluster data:
    | _get_cluster_Prot_Teff_data

Get and clean individual cluster data:
    | get_alphaPer
    | get_Pleiades
    | get_Blanco1
    | get_PscEri
    | get_NGC3532
    | get_GroupX
    | get_Praesepe_Rampalli_2021
    | get_NGC6811
    | get_NGC6819
    | get_Ruprecht147
    | get_M67
"""
#############
## LOGGING ##
#############
import logging
from gyrointerp import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############
import os
import numpy as np, pandas as pd
from astropy.table import Table
from astropy.io import fits
from copy import deepcopy
from matplotlib import cm

try:
    from cdips.utils.gaiaqueries import (
        given_dr3_sourceids_get_dr2_xmatch, given_source_ids_get_gaia_data,
        given_source_ids_get_neighbor_counts
    )
    HAVE_GAIA_UTILS = 1
except (ModuleNotFoundError, AssertionError) as e:
    # if cdips is not installed (github.com/lgbouma/cdips), and specifically if
    # the gaia credentials have not been correctly configured at
    # ~/.gaia_credentials per ``cdips.utils.gaiaqueries``, suppress the
    # associated error.
    LOGWARNING(e)
    HAVE_GAIA_UTILS = 0

from gyrointerp.paths import LOCALDIR, DATADIR, RESULTSDIR
from gyrointerp.extinctionpriors import extinction_A_V_dict
from gyrointerp.teff import (
    given_dr2_BpmRp_AV_get_Teff_Curtis2020, _given_VmKs_get_Teff,
    _given_GmKs_get_Teff
)

from gyrointerp.helpers import (
    prepend_colstr, left_merge, given_dr2_get_dr3_dataframes
)

def _get_cluster_Prot_Teff_data(N_colors=5, logo_colors=0):
    """
    This function retrieves dataframes of all available reference cluster data.
    It also defines the colors / labels / zorders used across many plots in the
    BPH23 manuscript.
    """
    assert N_colors in [5,6,7]

    overwrite = 0

    df_aper = get_alphaPer(overwrite=overwrite)
    df_psceri = get_PscEri(overwrite=overwrite)
    df_bla1 = get_Blanco1(overwrite=overwrite)
    df_plei = get_Pleiades(overwrite=overwrite)
    df_3532 = get_NGC3532(overwrite=overwrite)
    df_grpx = get_GroupX(overwrite=overwrite)
    df_prae = get_Praesepe_Rampalli_2021(overwrite=overwrite)
    df_6811 = get_NGC6811(overwrite=overwrite)
    df_6819 = get_NGC6819(overwrite=overwrite)
    df_r147 = get_Ruprecht147(overwrite=overwrite)
    df_m67 = get_M67(overwrite=overwrite)

    # tested, manuscript
    #cmap = cm.terrain(np.linspace(0,0.8,N_colors))
    #cmap = cm.Accent(np.linspace(0,0.6,N_colors))
    #cmap = cm.hsv(np.linspace(0,0.8,N_colors))
    #cmap = [f"C{ix}" for ix in range(N_colors)]
    #cmap = cm.Set3(np.linspace(0,0.5,N_colors))
    #cmap = cm.tab20c(np.linspace(0.1,1,N_colors))

    # adopted, manuscript
    cmap = ["#5CC8CB", "#2B64C6", "#F9E16A",
            "#81C83D", '#EC702D', '#E9A2AE',
            "k"]

    if logo_colors:
        # ADOPTED LOGO COLORS
        cmap = [None, "#ffb3ba", None, "#ffffba","#bae1ff", None]

    manual_colors = 0
    if manual_colors:
        cmap = cm.RdYlBu(np.linspace(0,1,N_colors))
        cmap = [None, cmap[0], None, cmap[1], cmap[2], None]

    # contents:
    # prot/teff dataframe, RGB color, label, zorder
    z0 = 2
    d = {
        'α Per': [df_aper, cmap[0], '80 Myr α Per', z0+12, "o"],
        'Pleiades': [df_plei, cmap[1], '120 Myr Pleiades', z0+2, "o"],
        'Blanco-1': [df_bla1, cmap[1], '120 Myr Blanco-1', z0+2, "8"],
        'Psc-Eri': [df_psceri, cmap[1], '120 Myr Psc-Eri', z0+2, "p"],
        'NGC-3532': [df_3532, cmap[2], '300 Myr NGC-3532', z0+4, "o"],
        'Group-X':  [df_grpx, cmap[2], '300 Myr Group-X', z0+4, "8"],
        'Praesepe': [df_prae, cmap[3], '670 Myr Praesepe', z0+6, "o"],
        'NGC-6811': [df_6811, cmap[4], '1 Gyr NGC-6811', z0+8, "o"],
        'NGC-6819': [df_6819, cmap[5], '2.5 Gyr NGC-6819', z0+10, "8"],
        'Ruprecht-147': [df_r147, cmap[5], '2.7 Gyr Rup-147', z0+10, "o"],
        'M67': [df_m67, cmap[6], '4 Gyr M67', z0+12, "o"],
        '120-Myr': [None, cmap[1], '', None, None],
        '300-Myr': [None, cmap[2], '', None, None],
        '2.6-Gyr': [None, cmap[5], '', None, None],
    }

    return d


def get_NGC6819(overwrite=0):
    """
    Return NGC-6819 (Meibom+2015), as processed by Curtis+2020 (Table 5).
    """

    cluster = 'ngc6819'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, "Curtis_2020_ngc6819_X_GDR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    fitspath = os.path.join(DATADIR, "literature",
                            "Curtis_2020_t5_composite_923_rows.fits")
    hdul = fits.open(fitspath)
    df = Table(hdul[1].data).to_pandas()
    hdul.close()

    # require finite Gaia DR2 source_id.  (Drop 2 rows of the 440, wide
    # binaries).
    df = df[df.Cluster == "NGC 6819"]

    dr2_source_ids = np.array(df.GaiaDR2).astype(np.int64)
    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, "Curtis_2020_ngc6819_DR2", "Curtis_2020_ngc6819_DR3",
        overwrite=overwrite
    )
    mdf = deepcopy(df)
    mdf = mdf.rename({'GaiaDR2':'dr2_source_id'}, axis='columns')

    mdf['dr2_bp_rp'] = mdf['BP-RP'] # gaia dr2 BP-RP

    # Merge
    mdf0 = left_merge(mdf, s_dr3, 'dr2_source_id', 'dr2_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["NGC-6819"]
    )

    dGmag = 3.25 # anything within 20x the brightness of the target
    sep_arcsec = 1.5*3.98 # and 1.5 Kepler pixels
    runid = "Curtis_2020_ngc6819_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert np.all(
        count_df.source_id.astype(str) == mdf.dr3_source_id.astype(str)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    # NOTE: CAMD and RV cuts already done by Curtis+2020, so omitting.

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_possible_binary"] = (
        (mdf.flag_ruwe_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
        |
        (mdf.flag_bad_gaiamatch)
    )

    # follow advice from
    # https://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/ApJ/904/140/table1&-out.max=50&-out.form=HTML%20Table&-oc.form=sexa
    # "Bench" column notes / definition
    mdf['flag_pass_author_quality'] = mdf.Prot > 0

    mdf["flag_benchmark_period"] = (
        mdf.flag_pass_author_quality
        #&
        #(~mdf.flag_possible_binary)
        &
        (~mdf.flag_bad_gaiamatch)
    )

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_Ruprecht147(overwrite=0):
    """
    Return Curtis+2020 (Table 1) rotation period dataframe with keys:
    "Prot", "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ the usual Gaia DR3 columns.

    NOTE:
        The "flag_possible_binary" flag in this case is weaker than in the
        other comparison clusters, because there aren't very many good stars.
    """

    cluster = 'ruprecht147'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, "Curtis_2020_rup147_X_GDR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    fitspath = os.path.join(DATADIR, "literature",
                            "Curtis_2020_t1_ruprecht147_440_rows.fits")
    hdul = fits.open(fitspath)
    df = Table(hdul[1].data).to_pandas()
    hdul.close()

    # require finite Gaia DR2 source_id.  (Drop 2 rows of the 440, wide
    # binaries).
    df = df[(~pd.isnull(df.GaiaDR2)) & (df.GaiaDR2 > 0)]

    dr2_source_ids = np.array(df.GaiaDR2).astype(np.int64)
    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, "Curtis_2020_rup147_DR2", "Curtis_2020_rup147_DR3",
        overwrite=overwrite
    )
    mdf = deepcopy(df)
    mdf = mdf.rename({'GaiaDR2':'dr2_source_id'}, axis='columns')
    mdf = mdf.drop(['ruwe'], axis='columns')

    mdf['dr2_bp_rp'] = mdf['BP-RP'] # gaia dr2 BP-RP

    # Merge
    mdf0 = left_merge(mdf, s_dr3, 'dr2_source_id', 'dr2_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["Ruprecht-147"]
    )

    dGmag = 3.25 # anything within 20x the brightness of the target
    sep_arcsec = 1.5*3.98 # and 1.5 Kepler pixels
    runid = "Curtis_2020_rup147_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert np.all(
        count_df.source_id.astype(str) == mdf.dr3_source_id.astype(str)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    # NOTE: CAMD and RV cuts already done by Curtis+2020, so omitting.

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_possible_binary"] = (
        (mdf.flag_ruwe_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
        |
        (mdf.flag_bad_gaiamatch)
    )

    mdf['flag_pass_author_quality'] = (
        (mdf.Bench == 'Yes')
        &
        (mdf.Prot > 0)
    )

    # follow advice from
    # https://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/ApJ/904/140/table1&-out.max=50&-out.form=HTML%20Table&-oc.form=sexa
    # "Bench" column notes / definition
    mdf["flag_benchmark_period"] = (
        (mdf.flag_pass_author_quality)
        &
        (~mdf.flag_possible_binary)
    )


    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_NGC6811(overwrite=0):
    """
    Returns the Curtis+2019 (Table 1) rotation period dataframe with keys:
    "Prot", "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ "flag_possible_binary", "flag_ruwe_outlier", "flag_camd_outlier",
    "flag_rverror_outlier", _plus_ the usual Gaia DR3 columns.
    """

    cluster = 'ngc6811'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, "Curtis_2019_X_GDR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    fitspath = os.path.join(DATADIR, "literature",
                            "Curtis_2019_table1_171rows.fits")
    hdul = fits.open(fitspath)
    df = Table(hdul[1].data).to_pandas()
    hdul.close()

    dr2_source_ids = np.array(df.Gaia).astype(np.int64)
    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, "Curtis_2019_ngc6811_DR2", "Curtis_2019_ngc6811_DR3",
        overwrite=overwrite
    )
    mdf = deepcopy(df)
    mdf = mdf.rename({'Gaia':'dr2_source_id'}, axis='columns')

    gcdf = given_source_ids_get_gaia_data(
        dr2_source_ids, "Curtis_2019_dr2_colors", n_max=10000,
        overwrite=overwrite, enforce_all_sourceids_viable=True,
        which_columns='g.source_id, g.bp_rp', gaia_datarelease='gaiadr2'
    )
    gcdf = gcdf.rename({"source_id":"dr2_source_id"}, axis='columns')
    assert np.all(gcdf.dr2_source_id.astype(str) == mdf.dr2_source_id.astype(str))
    mdf['dr2_bp_rp'] = gcdf['bp_rp']

    # Merge
    mdf0 = left_merge(mdf, s_dr3, 'dr2_source_id', 'dr2_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["Pleiades"]
    )

    dGmag = 3.25 # anything within 20x the brightness of the target
    sep_arcsec = 1.5*3.98 # and 1.5 Kepler pixels
    runid = "Curtis_2019_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert np.all(
        count_df.source_id.astype(str) == mdf.dr3_source_id.astype(str)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"camd_outliers.csv"
    )
    rverror_outliercsv = os.path.join(
        outdir, f"rverror_outliers.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        errmsg = (
            f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
            f'outliers in M_G vs bp_rp, g_rp, bp_g and write '
            f'to {camd_outliercsv}.  Do the same in radial_velocity_error '
            f'versus bp_rp, and save to {rverror_outliercsv}\n'
        )
        errmsg += 42*'-'
        LOGINFO(42*'-')
        raise AssertionError(errmsg)

    assert os.path.exists(camd_outliercsv)
    assert os.path.exists(rverror_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)
    df_rverror_outlier = pd.read_csv(rverror_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rverror_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rverror_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_possible_binary"] = (
        (mdf.flag_ruwe_outlier)
        |
        (mdf.flag_camd_outlier)
        |
        (mdf.flag_rverror_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
        |
        (mdf.flag_bad_gaiamatch)
    )

    mdf["flag_pass_author_quality"] = (
        (mdf.f_Seq == 'Y')
    )

    mdf["flag_benchmark_period"] = (
        (mdf.flag_pass_author_quality)
        &
        (~mdf.flag_possible_binary)
    )

    mdf = mdf.rename({"Per":"Prot"}, axis='columns')

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf



def get_Blanco1(overwrite=0):
    """
    Returns the Gillen+2020 (Table 1) rotation period dataframe with keys:
    "Prot", "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ "flag_possible_binary", "flag_ruwe_outlier", "flag_camd_outlier",
    "flag_rverror_outlier", _plus_ the usual Gaia DR3 columns.
    """

    cluster = 'blanco1'
    authoryr = "Gillen_2020"
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, f"{authoryr}_X_GDR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    csvpath = os.path.join(DATADIR, "literature",
                            "Gillen_2020_Blanco1_table1.dat")
    df = pd.read_csv(csvpath, delim_whitespace=True)

    dr2_source_ids = np.array(df.Gaia_ID).astype(np.int64)
    df['dr2_source_id'] = dr2_source_ids
    df['dr2_bp_rp'] = df['BP_mag'] - df['RP_mag']

    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, f"{authoryr}_{cluster}_DR2", f"{authoryr}_{cluster}_DR3",
        overwrite=overwrite
    )
    mdf = deepcopy(df)

    # Merge
    mdf0 = left_merge(mdf, s_dr3, 'dr2_source_id', 'dr2_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["Blanco-1"]
    )

    dGmag = 3.25 # anything within 20x the brightness of the target
    sep_arcsec = 1*21 # and 1 TESS pixel
    runid = f"{authoryr}_{cluster}_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert (
        np.all((count_df.source_id.astype(str) ==
                mdf.dr3_source_id.astype(str)))
        and
        np.all(count_df.index == mdf.index)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"camd_outlier.csv"
    )
    rverror_outliercsv = os.path.join(
        outdir, f"rverror_outlier.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        errmsg = (
            f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
            f'outliers in M_G vs bp_rp, g_rp, bp_g and write '
            f'to {camd_outliercsv}.  Do the same in radial_velocity_error '
            f'versus bp_rp, and save to {rverror_outliercsv}\n'
        )
        errmsg += 42*'-'
        LOGINFO(42*'-')
        raise AssertionError(errmsg)

    assert os.path.exists(camd_outliercsv)
    assert os.path.exists(rverror_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)
    df_rverror_outlier = pd.read_csv(rverror_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rverror_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rverror_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_possible_binary"] = (
        (mdf.flag_camd_outlier)
        |
        (mdf.flag_rverror_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
        |
        (mdf.flag_bad_gaiamatch)
    )

    mdf['flag_pass_author_quality'] = (
        ~mdf.mult.str.contains("r") # RV binaries from Gillen+20 table
    )

    mdf["flag_benchmark_period"] = (
        (~mdf.flag_possible_binary)
        &
        (mdf.flag_pass_author_quality)
    )

    mdf = mdf.rename({"P_adopt": "Prot"}, axis='columns')

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_Pleiades(overwrite=0):
    """
    Returns the Rebull+2016 (Table 2) rotation period dataframe with keys:
    "Prot", "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ "flag_possible_binary", "flag_ruwe_outlier", "flag_camd_outlier",
    "flag_rverror_outlier", _plus_ the usual Gaia DR3 columns.
    """

    cluster = 'pleiades'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, "Rebull_2016_X_GDR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    fitspath = os.path.join(DATADIR, "literature",
                            "Rebull_2016_Table2_759_stars.fits")
    hdul = fits.open(fitspath)
    df = Table(hdul[1].data).to_pandas()
    hdul.close()

    # Interpolate effective temperatures from the Mamajek table and (V-Ks)0.
    # (less reddening dependence than B-V).
    df['Teff_Mamajek'] = _given_VmKs_get_Teff(df['__V-K_0'].astype(float))

    # Crossmatch from EPIC --> DR2, using the 1" crossmatch from
    # https://gaia-kepler.fun/, downloaded to LOCALDIR
    # Keep matches with the closest K2 to Gaia DR2 angular distance as the
    # default best match.
    # This procedure misses five objects, presumably due to errors in
    # the proper motion propagation:
    # EPICs ['210966700', '211059981', '210903023', '220115919', '211128979']

    epic_dr2_path = os.path.join(LOCALDIR, "k2_dr2_1arcsec.fits")
    hdul = fits.open(epic_dr2_path)
    epic_df = Table(hdul[1].data).to_pandas()
    hdul.close()

    get_best_epic = lambda _df: (
            _df.sort_values(by='k2_gaia_ang_dist').
            drop_duplicates(subset='epic_number', keep='first')
    )
    epic_df = get_best_epic(epic_df)

    selcols = ['source_id', 'epic_number', 'tm_name', 'k2_gaia_ang_dist',
               'bp_rp']
    epic_df = epic_df[selcols]
    epic_df = epic_df.rename(
        {'source_id':'dr2_source_id', 'bp_rp':'dr2_bp_rp'}, axis='columns'
    )
    epic_df['dr2_source_id'] = epic_df.dr2_source_id.astype(str)

    mdf = left_merge(df, epic_df, 'EPIC', 'epic_number')

    N_missing = len(mdf[pd.isnull(mdf.dr2_source_id)])
    LOGINFO(f'Missing {N_missing}/{len(df)} in the Rebull16->GDR2 crossmatch.')
    LOGINFO('Dropping them!')

    mdf = mdf[~pd.isnull(mdf.dr2_source_id)]
    mdf['dr2_source_id'] = mdf.dr2_source_id.astype(str)
    assert len(mdf) == len(df) - N_missing

    dr2_source_ids = np.array(mdf.dr2_source_id).astype(np.int64)
    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, "Rebull_2016_DR2", "Rebull_2016_DR3_gaiadr3",
        overwrite=overwrite
    )

    # Merge
    mdf0 = left_merge(mdf, s_dr3, 'dr2_source_id', 'dr2_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["Pleiades"]
    )

    dGmag = 3.25 # anything within 100x the brightness of the target
    sep_arcsec = 1.5*3.98 # and 1.5 Kepler pixels
    runid = "Rebull_2016_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert np.all(
        count_df.source_id.astype(str) == mdf.dr3_source_id.astype(str)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    cluster = 'pleiades'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"camd_outlier.csv"
    )
    rverror_outliercsv = os.path.join(
        outdir, f"rverror_outlier.csv"
    )
    rv_outliercsv = os.path.join(
        outdir, f"rv_outlier.csv"
    )
    fdwarf_outliercsv = os.path.join(
        outdir, f"fdwarf_outlier.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        errmsg = (
            f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
            f'outliers in M_G vs bp_rp, g_rp, bp_g, V-Ks_0, and write '
            f'to {camd_outliercsv}.  Do the same in radial_velocity_error '
            f'versus bp_rp, and save to {rverror_outliercsv}\n'
        )
        errmsg += 42*'-'
        LOGINFO(42*'-')
        raise AssertionError(errmsg)

    assert os.path.exists(camd_outliercsv)
    assert os.path.exists(rverror_outliercsv)
    assert os.path.exists(rv_outliercsv)
    assert os.path.exists(fdwarf_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)
    df_rverror_outlier = pd.read_csv(rverror_outliercsv)
    df_rv_outlier = pd.read_csv(rv_outliercsv)
    df_fdwarf_outlier = pd.read_csv(fdwarf_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rverror_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rverror_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rv_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rv_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_fdwarf_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_fdwarf_outlier.dr3_source_id.astype(str)
    )

    # Benchmark Pleiades periods:
    # Include:
    # * "best" membership indicator from Rebull+2016
    # * stars with only a single reported period from Rebull+2016
    # Exclude:
    # * Bad gaia DR2 <-> DR3 matches (angular distance > 2 arcsec or magnitude
    #       difference > 0.1 mag)
    # * Stars with Gaia DR3 RVs >~ 30 km/s from the cluster mean
    # * The four binaries in the "F-dwarf clump" described by Stauffer+2016 in
    #       their section 5.1.
    # * RUWE > 1.2
    # * manually identified CAMD outliers (from M_G vs bp_rp, g_rp, bp_g, V-Ks_0)
    # * RV error outliers
    # * Things the Gaia DR3 pipeline flagged as non-single stars

    sel = mdf.Per2 == 0
    mdf.loc[sel, 'Per2'] = np.nan

    mdf["flag_pass_author_quality"] = (
        (~mdf.flag_fdwarf_outlier)
        &
        (pd.isnull(mdf.Per2)) # multiple photometric periods usually implies binarity
        &
        (mdf.Mm == 'best')
    )

    mdf["flag_possible_binary"] = (
        (mdf.flag_ruwe_outlier)
        |
        (mdf.flag_camd_outlier)
        |
        (mdf.flag_rverror_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
        |
        (mdf.flag_bad_gaiamatch)
        |
        (mdf.flag_rv_outlier)
    )

    mdf["flag_benchmark_period"] = (
        (mdf.flag_pass_author_quality)
        &
        (~mdf.flag_possible_binary)
    )

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_NGC3532(overwrite=0):
    """
    Returns the Fritzewski+2021 rotation period dataframe with keys:
    "Prot", "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ "flag_possible_binary", "flag_ruwe_outlier", "flag_camd_outlier",
    "flag_rverror_outlier", _plus_ the usual Gaia DR3 columns.
    """

    cluster = 'ngc3532'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, "Fritzewski_2021_X_DR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    fitspath = os.path.join(DATADIR, "literature",
                            "Fritzewski_2021_AA_652_60_table2_NGC3532.fits")
    hdul = fits.open(fitspath)
    df = Table(hdul[1].data).to_pandas()
    hdul.close()

    # Interpolate effective temperatures from the Mamajek table and (V-Ks)0.
    # (less reddening dependence than B-V).
    df['Teff_Mamajek'] = _given_VmKs_get_Teff(df['__V-Ks_0'].astype(float))

    dr2_source_ids = np.array(df.GaiaDR2).astype(np.int64)
    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, "Fritzewski_2021_DR2", "Fritzewski_2021_DR3_gaiadr3",
        overwrite=overwrite
    )

    gcdf = given_source_ids_get_gaia_data(
        dr2_source_ids, "Fritzewski_2021_dr2_colors", n_max=10000,
        overwrite=overwrite, enforce_all_sourceids_viable=True,
        which_columns='g.source_id, g.bp_rp', gaia_datarelease='gaiadr2'
    )
    gcdf = gcdf.rename({"source_id":"dr2_source_id"}, axis='columns')
    assert np.all(gcdf.dr2_source_id.astype(str) == df.GaiaDR2.astype(str))
    df['dr2_bp_rp'] = gcdf['bp_rp']

    # Merge
    mdf = left_merge(df, s_dr3, 'GaiaDR2', 'dr2_source_id')
    mdf = left_merge(mdf, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["NGC-3532"]
    )

    dGmag = 3.25 # anything within 100x the brightness of the target
    sep_arcsec = 1.5*0.289 # and 1.5 Y4KCam pixels (sec 2.1 of the paper)
    runid = "Fritzewski_2021_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert np.all(
        count_df.source_id.astype(str) == mdf.dr3_source_id.astype(str)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"{cluster}_camd_outliers.csv"
    )
    rverror_outliercsv = os.path.join(
        outdir, f"{cluster}_rverror_outliers.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        LOGINFO(42*'-')
        LOGINFO(f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
              'outliers in M_G vs bp_rp, g_rp, bp_g, V-Ks_0, and write '
              'to {camd_outliercsv}.  Do the same in radial_velocity_error '
              'versus bp_rp, and save to {rverror_outliercsv}')
        LOGINFO(42*'-')
        raise AssertionError

    assert os.path.exists(camd_outliercsv)
    assert os.path.exists(rverror_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)
    df_rverror_outlier = pd.read_csv(rverror_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rverror_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rverror_outlier.dr3_source_id.astype(str)
    )

    # Benchmark NGC-3532 periods:
    # Include:
    # * "First class" periods
    # * "algorithmic" periods.
    # Exclude:
    # * RUWE > 1.2
    # * manually identified CAMD outliers (from M_G vs bp_rp, g_rp, bp_g, V-Ks_0)
    # * RV error outliers (there are three)
    # * Things the Gaia DR3 pipeline flagged as non-single stars (there is
    # one).
    mdf["flag_possible_binary"] = (
        (mdf.flag_ruwe_outlier)
        |
        (mdf.flag_camd_outlier)
        |
        (mdf.flag_rverror_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
    )

    mdf["flag_pass_author_quality"] = (
        #(mdf.Class == 3) # include "activity informed" periods
        #|
        (mdf.Class == 1)
        |
        (mdf.Class == 2)
    )

    mdf["flag_benchmark_period"] = (
        (mdf.flag_pass_author_quality)
        &
        (~mdf.flag_possible_binary)
    )

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_Praesepe_Rampalli_2021(overwrite=0):
    """
    Returns the Rampalli+2021 Praesepe rotation period dataframe with keys:
    "Prot", "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ "flag_possible_binary", "flag_ruwe_outlier", "flag_camd_outlier",
    "flag_rverror_outlier", _plus_ the usual Gaia DR3 columns.
    """

    cluster = 'praesepe_rampalli2021'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, "Rampalli_2021_X_DR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(
            cachepath, dtype={'dr2_source_id':str, 'dr3_source_id':str}
        )

    table_path = os.path.join(DATADIR, "literature",
                              "Rampalli_2021_apjac0c1et3_mrt.txt")
    t = Table.read(table_path, format='cds')
    df_t3 = t.to_pandas()

    table_path = os.path.join(DATADIR, "literature",
                              "Rampalli_2021_apjac0c1et4_mrt.txt")
    t = Table.read(table_path, format='cds')
    df_t4 = t.to_pandas()

    # drop 17 cases without any assessment of previous rotation period quality
    df_t4 = df_t4[~pd.isnull(df_t4.QFClean)]
    assert len(df_t3) == len(df_t4)

    # get all the rotation period data and the star data in the same dataframe.
    # unsurprisingly, a few columns are duplicated.  In such cases, treat
    # "table 3" as the default ones.
    df = left_merge(df_t3, df_t4, 'Gaia', 'Gaia')
    df = df_t3.merge(
        df_t4, left_on="Gaia", right_on="Gaia", how='left',
        suffixes=("","_t4")
    )

    assert len(df) == len(df_t3)

    dr3_source_ids = np.array(
        df.Gaia.str.extract(r'(Gaia EDR3 )(\d*)')[1]
    ).astype(str)
    df['dr3_source_id'] = dr3_source_ids
    mdf = df

    dr3_source_ids = dr3_source_ids.astype(np.int64)

    runid_dr3 = 'Rampalli_2021_Praesepe_DR3'
    gdf = given_source_ids_get_gaia_data(
        dr3_source_ids, runid_dr3, n_max=10000, overwrite=overwrite,
        enforce_all_sourceids_viable=True, which_columns='*',
        gaia_datarelease='gaiadr3'
    )
    gdf = gdf.rename({"source_id":"dr3_source_id"}, axis='columns')

    selcols = ['dr3_source_id', 'ra', 'dec', 'parallax', 'parallax_error',
               'parallax_over_error', 'pmra', 'pmdec', 'ruwe',
               'phot_g_mean_flux_over_error', 'phot_g_mean_mag',
               'phot_bp_mean_flux_over_error', 'phot_rp_mean_flux',
               'phot_rp_mean_flux_over_error', 'phot_bp_rp_excess_factor',
               'phot_bp_n_contaminated_transits', 'phot_bp_n_blended_transits',
               'phot_rp_n_contaminated_transits', 'phot_rp_n_blended_transits',
               'bp_rp', 'bp_g', 'g_rp', 'radial_velocity',
               'radial_velocity_error', 'rv_method_used',
               'rv_expected_sig_to_noise', 'vbroad', 'vbroad_error', 'l', 'b',
               'ecl_lon', 'ecl_lat', 'non_single_star', 'teff_gspphot',
               'teff_gspphot_lower', 'teff_gspphot_upper', 'azero_gspphot',
               'ag_gspphot', 'ebpminrp_gspphot']

    gdf = gdf[selcols]

    #
    # Get the Gaia DR2 BP-RP color
    #
    dr2_x_dr3_df = given_dr3_sourceids_get_dr2_xmatch(
        dr3_source_ids, "Rampalli_2021_dr3_to_dr2", overwrite=overwrite,
        enforce_all_sourceids_viable=True
    )
    # Take the closest magnitude difference as the single match.
    get_dr2_xm = lambda _df: (
            _df.sort_values(by='abs_magnitude_difference').
            drop_duplicates(subset='dr3_source_id', keep='first')
    )
    s_dr2 = get_dr2_xm(dr2_x_dr3_df)
    LOGINFO(10*'-')
    LOGINFO(s_dr2.describe())
    LOGINFO(10*'-')

    dr2_source_ids = np.array(s_dr2.dr2_source_id).astype(np.int64)
    gcdf = given_source_ids_get_gaia_data(
        dr2_source_ids, "Rampalli_2021_dr2_color", n_max=10000,
        overwrite=overwrite, enforce_all_sourceids_viable=True,
        which_columns='g.source_id, g.bp_rp', gaia_datarelease='gaiadr2'
    )
    gcdf = gcdf.rename(
        {"source_id":"dr2_source_id", "bp_rp":"dr2_bp_rp"}, axis='columns'
    )
    assert np.all(
        np.array(gcdf.dr2_source_id) == np.array(s_dr2.dr2_source_id)
    )
    s_dr2 = s_dr2.merge(
        gcdf, on='dr2_source_id', how='left'
    )
    s_dr2['dr2_source_id'] = s_dr2['dr2_source_id'].astype(str)

    # Merge
    mdf0 = left_merge(mdf, s_dr2, 'dr3_source_id', 'dr3_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["Praesepe"]
    )

    dGmag = 3.25 # anything within 100x the brightness of the target
    sep_arcsec = 1.5*3.98 # and 1.5 Kepler pixels
    runid = "Rampalli_2021_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert np.all(
        count_df.source_id.astype(str) == mdf.dr3_source_id.astype(str)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"camd_outlier.csv"
    )
    rverror_outliercsv = os.path.join(
        outdir, f"rverror_outlier.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        LOGINFO(42*'-')
        LOGINFO(f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
              'outliers in M_G vs bp_rp, g_rp, bp_g, V-Ks_0, and write '
              'to {camd_outliercsv}.  Do the same in radial_velocity_error '
              'versus bp_rp, and save to {rverror_outliercsv}')
        LOGINFO(42*'-')
        raise AssertionError

    assert os.path.exists(camd_outliercsv)
    assert os.path.exists(rverror_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)
    df_rverror_outlier = pd.read_csv(rverror_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rverror_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rverror_outlier.dr3_source_id.astype(str)
    )

    # Benchmark periods:
    # Include:
    # * Clean quality flag only
    # Exclude:
    # * All possible signs of binarity
    # * Any missing EDR3 measurements.

    mdf["flag_possible_binary"] = (
        (mdf.flag_ruwe_outlier)
        |
        (mdf.flag_camd_outlier)
        |
        (mdf.flag_rverror_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
    )

    mdf["flag_pass_author_quality"] = (
        ~(
            (mdf.Multi == 1)
            |
            (mdf.Neigh == 1)
            |
            (mdf.Bin == 1)
            |
            (mdf.Binary == 1)
            |
            (mdf.Flag) # missing Gaia EDR3 measurement
        )
        &
        (mdf.QFClean == 1)
    )

    mdf["flag_benchmark_period"] = (
        (~mdf.flag_possible_binary)
        &
        (mdf.flag_pass_author_quality)
    )

    mdf['dr2_source_id'] = mdf.dr2_source_id.astype(str)
    mdf = mdf.drop({'source_id'}, axis='columns')

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_PscEri(overwrite=0):
    """
    Returns the Curtis+2019 (Table 2) rotation period dataframe with keys:
    "Prot", "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ "flag_possible_binary", "flag_ruwe_outlier", "flag_camd_outlier",
    "flag_rverror_outlier", _plus_ the usual Gaia DR3 columns.
    """

    cluster = 'psc-eri'
    authoryr = "Curtis_2019"
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, f"{authoryr}_X_GDR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    fitspath = os.path.join(DATADIR, "literature",
                            "Curtis_2019_PscEri_table2_101rows.fits")
    hdul = fits.open(fitspath)
    df = Table(hdul[1].data).to_pandas()
    hdul.close()

    dr2_source_ids = np.array(df.Source).astype(np.int64)
    df['dr2_source_id'] = dr2_source_ids

    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, f"{authoryr}_{cluster}_DR2", f"{authoryr}_{cluster}_DR3",
        overwrite=overwrite
    )
    mdf = deepcopy(df)
    mdf['dr2_bp_rp'] = mdf['GBP-GRP']

    # Merge
    mdf0 = left_merge(mdf, s_dr3, 'dr2_source_id', 'dr2_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["Psc-Eri"]
    )

    dGmag = 3.25 # anything within 20x the brightness of the target
    sep_arcsec = 1*21 # and 1 TESS pixel
    runid = f"{authoryr}_{cluster}_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert (
        np.all((count_df.source_id.astype(str) ==
                mdf.dr3_source_id.astype(str)))
        and
        np.all(count_df.index == mdf.index)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    mdf = mdf.drop(['Simbad'], axis='columns')

    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"camd_outliers.csv"
    )
    rverror_outliercsv = os.path.join(
        outdir, f"rverror_outliers.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        errmsg = (
            f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
            f'outliers in M_G vs bp_rp, g_rp, bp_g and write '
            f'to {camd_outliercsv}.  Do the same in radial_velocity_error '
            f'versus bp_rp, and save to {rverror_outliercsv}\n'
        )
        errmsg += 42*'-'
        LOGINFO(42*'-')
        raise AssertionError(errmsg)

    assert os.path.exists(camd_outliercsv)
    assert os.path.exists(rverror_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)
    df_rverror_outlier = pd.read_csv(rverror_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rverror_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rverror_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_possible_binary"] = (
        (mdf.flag_camd_outlier)
        |
        (mdf.flag_rverror_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
        |
        (mdf.flag_bad_gaiamatch)
    )

    mdf['flag_pass_author_quality'] = True

    mdf["flag_benchmark_period"] = (
        (~mdf.flag_possible_binary)
        &
        (mdf.flag_pass_author_quality)
    )

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_GroupX(overwrite=0):
    """
    Returns the Messina+2022 rotation period dataframe with keys: "Prot",
    "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ "flag_possible_binary", "flag_ruwe_outlier", "flag_camd_outlier",
    "flag_rverror_outlier", _plus_ the usual Gaia DR3 columns.
    """

    cluster = 'groupx'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, "Messina_2022_X_GDR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    fitspath = os.path.join(DATADIR, "literature",
                            "Messina_2022_AAP_groupx_173_stars.fits")
    hdul = fits.open(fitspath)
    df = Table(hdul[1].data).to_pandas()
    hdul.close()

    tic_ids = np.array(df.TIC)

    dr2_source_ids = []
    N_stars = len(tic_ids)
    from astrobase.services.identifiers import tic_to_gaiadr2
    for ix, tic_id in enumerate(tic_ids):
        LOGINFO(f"{ix}/{N_stars}...")
        dr2_source_ids.append(tic_to_gaiadr2(tic_id))
    assert len(dr2_source_ids) == len(tic_ids)

    dr2_source_ids = np.array(dr2_source_ids).astype(np.int64)
    df['dr2_source_id'] = dr2_source_ids

    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, "Messina_2022_groupx_DR2", "Messina_2022_groupx_DR3",
        overwrite=overwrite
    )
    mdf = deepcopy(df)

    gcdf = given_source_ids_get_gaia_data(
        dr2_source_ids, "Messina_2022_dr2_color", n_max=10000,
        overwrite=overwrite, enforce_all_sourceids_viable=True,
        which_columns='g.source_id, g.bp_rp', gaia_datarelease='gaiadr2'
    )
    gcdf = gcdf.rename(
        {"source_id":"dr2_source_id", "bp_rp":"dr2_bp_rp"}, axis='columns'
    )

    assert np.all(np.array(gcdf.dr2_source_id) == np.array(mdf.dr2_source_id))

    mdf['dr2_bp_rp'] = gcdf['dr2_bp_rp']

    # Merge
    mdf0 = left_merge(mdf, s_dr3, 'dr2_source_id', 'dr2_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["Group-X"]
    )
    mdf['Teff_Mamajek'] = _given_GmKs_get_Teff(mdf['__G-K_0'])

    dGmag = 3.25 # anything within 20x the brightness of the target
    sep_arcsec = 1*21 # and 1 TESS pixel
    runid = "Messina_2022_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert np.all(
        count_df.source_id.astype(str) == mdf.dr3_source_id.astype(str)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"camd_outliers.csv"
    )
    rverror_outliercsv = os.path.join(
        outdir, f"rverror_outliers.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        errmsg = (
            f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
            f'outliers in M_G vs bp_rp, g_rp, bp_g and write '
            f'to {camd_outliercsv}.  Do the same in radial_velocity_error '
            f'versus bp_rp, and save to {rverror_outliercsv}\n'
        )
        errmsg += 42*'-'
        LOGINFO(42*'-')
        raise AssertionError(errmsg)

    assert os.path.exists(camd_outliercsv)
    assert os.path.exists(rverror_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)
    df_rverror_outlier = pd.read_csv(rverror_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rverror_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rverror_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_possible_binary"] = (
        (mdf.flag_ruwe_outlier)
        |
        (mdf.flag_camd_outlier)
        |
        (mdf.flag_rverror_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
        |
        (mdf.flag_bad_gaiamatch)
    )

    mdf["flag_pass_author_quality"] = (
        (mdf.Grade == 'A')
        &
        (mdf.n_Seq == '  ')
    )

    mdf["flag_benchmark_period"] = (
        (mdf.flag_pass_author_quality)
        &
        (~mdf.flag_possible_binary)
    )

    mdf = mdf.rename({"Per":"Prot"}, axis='columns')

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_alphaPer(overwrite=0):
    """
    Returns the Boyle & Bouma 2023 rotation period dataframe with keys: "Prot",
    "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration),
    _plus_ "flag_possible_binary", "flag_ruwe_outlier", "flag_camd_outlier",
    "flag_rverror_outlier", _plus_ the usual Gaia DR3 columns.
    """

    # NOTE: overwrite has no effect for alpha-Per CSV.

    cluster = 'alpha-per'
    authoryr = "Boyle_table3_full"
    outdir = os.path.join(DATADIR, "interim", cluster)
    cachepath = os.path.join(outdir, f"{authoryr}_DR3_supp.csv")

    if os.path.exists(cachepath):
        LOGINFO(f"Found {cachepath}; returning.")
        return pd.read_csv(cachepath)

    df0 = pd.read_csv(os.path.join(
        outdir, "Boyle_table3_full.csv")
    )
    df1 = pd.read_csv(os.path.join(
        outdir, "Boyle_inprep_X_GDR3_supplemented.csv")
    )

    core_csv_path = os.path.join(outdir, "core_select_vl_vb_X_Y_Z.csv")
    if os.path.exists(core_csv_path):
        # hand-tuned core selection to match Boyle+Bouma 2023, for purposes of
        # the appendix figure.  these cuts were performed in the
        # core_select_vl_vb_X_Y_Z.glu glue session, to match the vl/vb/X/Y/Z
        # cuts described in the aforementioned manuscript.
        df2 = pd.read_csv(core_csv_path)
    else:
        df2 = None

    df = df0.merge(df1, how='left', left_on='dr3_source_id',
                   right_on='dr3_source_id', suffixes=("", "_supp"))

    if df2 is not None:
        in_core = df.dr3_source_id.isin(df2.dr3_source_id)
        df['flag_in_core'] = in_core

    df = df.rename(
        {'Teff_Curtis20':'Teff_Curtis20_fixedreddening'},
        axis='columns'
    )
    df = df.rename({
        "period":"Prot",
        "flag_benchmark_period":"flag_benchmark_period_Andy",
        "teff_curtis20":"Teff_Curtis20" #variable reddening,
    }, axis='columns')

    df['flag_pass_author_quality'] = df['in_gyro_sample']

    df['flag_benchmark_period'] = df['flag_pass_author_quality']

    df.to_csv(cachepath, index=False)

    LOGINFO(f"Wrote {cachepath}; returning.")
    return df


def get_alphaPer_construct(overwrite=0):
    """
    Constructor for ``getters.get_alphaPer``
    """

    cluster = 'alpha-per'
    authoryr = "Boyle_inprep"
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, f"{authoryr}_X_GDR3_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(cachepath)

    csvpath = os.path.join(DATADIR, "literature",
                            "Boyle_in_prep_alpha-Per.csv")
    df = pd.read_csv(csvpath)

    dr2_source_ids = np.array(df.dr2_source_id).astype(np.int64)
    df['dr2_source_id'] = dr2_source_ids

    gdf, s_dr3 = given_dr2_get_dr3_dataframes(
        dr2_source_ids, f"{authoryr}_{cluster}_DR2", f"{authoryr}_{cluster}_DR3",
        overwrite=overwrite
    )
    mdf = deepcopy(df)

    gcdf = given_source_ids_get_gaia_data(
        dr2_source_ids, f"{authoryr}_dr2_color", n_max=10000,
        overwrite=overwrite, enforce_all_sourceids_viable=True,
        which_columns='g.source_id, g.bp_rp', gaia_datarelease='gaiadr2'
    )
    gcdf = gcdf.rename(
        {"source_id":"dr2_source_id", "bp_rp":"dr2_bp_rp"}, axis='columns'
    )

    # careful with this... better to actually just .merge!
    assert (
        np.all(np.array(gcdf.dr2_source_id) == np.array(mdf.dr2_source_id))
        and
        np.all(gcdf.index == mdf.index)
    )

    mdf['dr2_bp_rp'] = gcdf['dr2_bp_rp']

    # Merge
    mdf0 = left_merge(mdf, s_dr3, 'dr2_source_id', 'dr2_source_id')
    mdf = left_merge(mdf0, gdf, 'dr3_source_id', 'dr3_source_id')

    mdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(mdf.dr2_bp_rp), extinction_A_V_dict["alpha-Per"]
    )

    dGmag = 3.25 # anything within 20x the brightness of the target
    sep_arcsec = 1*21 # and 1 TESS pixel
    runid = f"{authoryr}_{cluster}_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert (
        np.all((count_df.source_id.astype(str) ==
                mdf.dr3_source_id.astype(str)))
        and
        np.all(count_df.index == mdf.index)
    )

    mdf['nbhr_count'] = count_df['nbhr_count']
    mdf['flag_nbhr_count'] = (
        mdf['nbhr_count'] >= 1
    )

    if "abs_magnitude_difference" not in mdf:
        mdf["abs_magnitude_difference"] = np.abs(mdf.magnitude_difference)

    mdf["flag_bad_gaiamatch"] = (
        (mdf.angular_distance > 2)
        |
        (mdf.abs_magnitude_difference > 0.2)
    )

    mdf["M_G"] = mdf['phot_g_mean_mag'] + 5*np.log10(mdf['parallax']/1e3) + 5

    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"camd_outliers.csv"
    )
    rverror_outliercsv = os.path.join(
        outdir, f"rv_error_outliers.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        errmsg = (
            f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
            f'outliers in M_G vs bp_rp, g_rp, bp_g and write '
            f'to {camd_outliercsv}.  Do the same in radial_velocity_error '
            f'versus bp_rp, and save to {rverror_outliercsv}\n'
        )
        errmsg += 42*'-'
        LOGINFO(42*'-')
        raise AssertionError(errmsg)

    assert os.path.exists(camd_outliercsv)
    assert os.path.exists(rverror_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)
    df_rverror_outlier = pd.read_csv(rverror_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_rverror_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_rverror_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_possible_binary"] = (
        (mdf.flag_camd_outlier)
        |
        (mdf.flag_rverror_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
    )

    mdf["flag_benchmark_period"] = (
        (~mdf.flag_possible_binary)
        &
        (~mdf.flag_bad_gaiamatch)
    )

    mdf = mdf.drop(['Unamed: 0'], axis='columns')

    mdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return mdf


def get_M67(overwrite=0):
    """
    Return concatenation of Barnes+2016, Dungee+2022, and Gruner+2023 M67
    sequences, with keys:
    "Prot", "Teff_Curtis20", "flag_benchmark_period" (for gyro calibration).
    """

    cluster = 'M67'
    outdir = os.path.join(DATADIR, "interim", cluster)
    if not os.path.exists(outdir): os.mkdir(outdir)
    cachepath = os.path.join(outdir, "M67_supplemented.csv")

    if os.path.exists(cachepath) and not overwrite:
        LOGINFO(f"Found {cachepath}, and not overwrite; returning.")
        return pd.read_csv(
            cachepath, dtype={'dr2_source_id':str, 'dr3_source_id':str}
        )

    table_path = os.path.join(
        DATADIR, "literature", "Dungee_2022_apjac90bet2_mrt.txt"
    )
    t = Table.read(table_path, format='cds')
    df_d22 = t.to_pandas()
    df_d22 = df_d22[df_d22.converged == 'True']
    df_d22['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(df_d22["gaiaBPmag"]) - np.array(df_d22["gaiaRPmag"]),
        extinction_A_V_dict['M67']
    )

    csvpath = os.path.join(
        DATADIR, "literature", "Gruner_Barnes_Weingrill_2023_M67_tableC2.csv"
    )
    df_g23 = pd.read_csv(csvpath)
    df_g23['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(df_g23["(BP-RP)0"]), 0
    )

    csvpath = os.path.join(
        DATADIR, "literature",
        "Barnes_2016_apj523472t1_ascii_X_DR2_supplemented.txt"
    )
    df_b16 = pd.read_csv(csvpath)
    df_b16['Teff_Curtis20'] =  given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(df_b16["bp_rp"]), extinction_A_V_dict['M67']
    )

    DEBUG_PLOT = 1
    if DEBUG_PLOT:
        from gyrointerp.models import slow_sequence
        import matplotlib.pyplot as plt
        plt.close("all")
        fig,ax = plt.subplots()
        ax.scatter(df_d22['Teff_Curtis20'], df_d22['prot'])
        ax.scatter(df_g23['Teff_Curtis20'], df_g23['period'])
        ax.scatter(df_b16['Teff_Curtis20'], df_b16['P'])
        _teffs = np.linspace(3800,6200,500)
        ax.plot(
            _teffs, slow_sequence(_teffs, 4000), c='k', zorder=-1, lw=0.5
        )
        ax.update({
            'xlim': ax.get_xlim()[::-1]
        })
        outpath = os.path.join(RESULTSDIR, "debug", "M67_lit_data.png")
        fig.savefig(outpath, bbox_inches='tight')

    df_d22 = df_d22.rename({'prot':'Prot'}, axis='columns')
    df_g23 = df_g23.rename({'period':'Prot'}, axis='columns')
    df_b16 = df_b16.rename({'P':'Prot'}, axis='columns')

    outdf = pd.concat((
        df_b16[['Teff_Curtis20', 'Prot']],
        df_d22[['Teff_Curtis20', 'Prot']],
        df_g23[['Teff_Curtis20', 'Prot']]
    ))
    # This seems to be concensus from Gruner+2023 and Dungee+2022.
    outdf['flag_benchmark_period'] = (
        (outdf.Prot > 12)
        &
        (outdf.Prot < 60)
    )

    outdf.to_csv(cachepath, index=False)
    LOGINFO(f"Wrote {cachepath}")

    return outdf
