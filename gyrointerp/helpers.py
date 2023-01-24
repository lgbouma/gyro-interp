"""
This module contains reusable helper functions.  The most generally useful one
will be ``get_summary_statistics``.
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
from os.path import join
import numpy as np, pandas as pd

def get_summary_statistics(age_grid, age_post, N=int(1e5)):
    """
    Given an age posterior probability density, ``age_post``, over a grid over
    ages, ``age_grid``, determine summary statistics for the posterior (its
    median, mean, +/-1 and 2-sigma intervals, etc).  Do this by sampling `N`
    times from the posterior, with replacement, while weighting by the
    probability.

    Args:

        age_grid (np.ndarray):
            Array-like of ages, in units of megayears.  For instance, the
            default *age_grid* in ``gyro_posterior.gyro_age_posterior`` is
            ``np.linspace(0, 3000, 500)``.

        age_post (np.ndarray):
            Posterior probability distribution for ages; length should match
            *age_grid*.  The posterior probabilities returned
            by``gyro_posterior.gyro_age_posterior`` and
            ``gyro_posterior.gyro_age_posterior_list`` are examples that would
            work.  Generally, this helper function works for any grid and
            probability distribution.


    Returns:

        dict : summary_statistics

            Dictionary containing keys and values for median, mean, peak
            (mode), +/-1sigma, +/-2sigma, +/-3sigma, and +/-1sigmapct.  The
            units of all values are megayears, except for *+/-1sigmapct*, which
            is the relative +/-1-sigma uncertainty normalized by the median of
            the posterior, and is dimensionless.
    """
    # This function is a thin wrapper to _given_grid_post_get_summary_statistics
    return _given_grid_post_get_summary_statistics(
        age_grid, age_post, N=N
    )


def _given_grid_post_get_summary_statistics(age_grid, age_post, N=int(1e5)):

    age_peak = int(age_grid[np.argmax(age_post)])

    df = pd.DataFrame({'age':age_grid, 'p':age_post})
    try:
        sample_df = df.sample(n=N, replace=True, weights=df.p)
    except ValueError:
        outdict = {
            'median': np.nan,
            'peak': np.nan,
            'mean': np.nan,
            '+1sigma': np.nan,
            '-1sigma': np.nan,
            '+2sigma': np.nan,
            '-2sigma': np.nan,
            '+3sigma': np.nan,
            '-3sigma': np.nan,
            '+1sigmapct': np.nan,
            '-1sigmapct': np.nan,
        }
        return outdict

    one_sig = 68.27/2
    two_sig = 95.45/2
    three_sig = 99.73/2

    pct_50 = np.nanpercentile(sample_df.age, 50)

    p1sig = np.nanpercentile(sample_df.age, 50+one_sig) - pct_50
    m1sig = pct_50 - np.nanpercentile(sample_df.age, 50-one_sig)

    p2sig = np.nanpercentile(sample_df.age, 50+two_sig) - pct_50
    m2sig = pct_50 - np.nanpercentile(sample_df.age, 50-two_sig)

    p3sig = np.nanpercentile(sample_df.age, 50+three_sig) - pct_50
    m3sig = pct_50 - np.nanpercentile(sample_df.age, 50-three_sig)

    outdict = {
        'median': np.round(pct_50,2),
        'peak': np.round(age_peak,2),
        'mean': np.round(np.nanmean(sample_df.age),2),
        '+1sigma': np.round(p1sig,2),
        '-1sigma': np.round(m1sig,2),
        '+2sigma': np.round(p2sig,2),
        '-2sigma': np.round(m2sig,2),
        '+3sigma': np.round(p3sig,2),
        '-3sigma': np.round(m3sig,2),
        '+1sigmapct': np.round(p1sig/pct_50,2),
        '-1sigmapct': np.round(m1sig/pct_50,2),
    }

    return outdict


def prepend_colstr(colstr, df):
    # prepend a string, `colstr`, to all columns in a dataframe
    return df.rename(
        {c:colstr+c for c in df.columns}, axis='columns'
    )


def left_merge(df0, df1, col0, col1):
    # execute a left-join ensuring the columns are cast as strings
    df0[col0] = df0[col0].astype(str)
    df1[col1] = df1[col1].astype(str)
    return df0.merge(
        df1, left_on=col0, right_on=col1, how='left'
    )

def given_dr2_get_dr3_dataframes(dr2_source_ids, runid_dr2, runid_dr3,
                                 overwrite=False):
    # dr2_source_ids: np.ndarray of np.int64 Gaia DR2 source identifiers.
    # runid_dr2: arbitrary string to identify the DR2->DR3 xmatch query
    # runid_dr3: arbitrary (different) string to identify the DR3 query

    # pip install cdips
    from cdips.utils.gaiaqueries import (
        given_dr2_sourceids_get_edr3_xmatch, given_source_ids_get_gaia_data
    )

    LOGINFO(42*'-')
    LOGINFO(runid_dr2)

    # Crossmatch from Gaia DR2->DR3.
    dr2_x_dr3_df = given_dr2_sourceids_get_edr3_xmatch(
        dr2_source_ids, runid_dr2, overwrite=overwrite,
        enforce_all_sourceids_viable=True
    )
    # Take the closest magnitude difference as the single match.
    #
    # In NGC-3532 case, yields matches for everything, largest angular distance
    # 1.3 arcseconds, largest magnitude difference G=0.06 mags.
    #
    # For Pleiades, trickier, since the sample goes fainter.  Lack of proper
    # motion projection also leads to many errorneous cases.
    get_dr3_xm = lambda _df: (
            _df.sort_values(by='abs_magnitude_difference').
            drop_duplicates(subset='dr2_source_id', keep='first')
    )
    s_dr3 = get_dr3_xm(dr2_x_dr3_df)
    LOGINFO(10*'-')
    LOGINFO(s_dr3.describe())
    LOGINFO(10*'-')
    if len(s_dr3) != len(np.unique(dr2_source_ids)):
        LOGINFO('Got bad dr2<->dr3 match')
        LOGINFO(len(s_dr3), len(np.unique(dr2_source_ids)))
        raise AssertionError

    assert len(s_dr3) == len(np.unique(dr2_source_ids))

    dr3_source_ids = np.array(s_dr3.dr3_source_id).astype(np.int64)

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

    selcols =['dr2_source_id', 'dr3_source_id', 'angular_distance',
              'magnitude_difference']
    s_dr3 = s_dr3[selcols]

    return gdf, s_dr3


def get_population_hyperparameter_posterior_samples():
    """
    Access the posterior samples described in section 3.5 of BPH23.
    (These are generated by ``drivers.run_emcee_fit_gyro_model``).

    The returned numpy array is samples in the following parameters: a1/a0,
    y_g, logk0, logk1, log_f.

    The notation follows Sections 3.3-3.5 of BPH23.
    """

    from gyrointerp.paths import CACHEDIR

    csvpath = join(CACHEDIR, "fit_120-Myr_300-Myr_Praesepe.csv.gz")
    if not os.path.exists(csvpath):
        # Pull the population-level hyperparameters from an external cache if
        # they are not already downloaded.
        dropboxlink = (
            'https://www.dropbox.com/s/ywe3z8ez2ll871m/fit_120-Myr_300-Myr_Praesepe.csv?dl=1'
        )
        df = pd.read_csv(dropboxlink)
        df.to_csv(csvpath, index=False)
        LOGINFO(f"Downloaded {csvpath} and cached it locally.")
    else:
        df = pd.read_csv(csvpath)
    flat_samples = np.array(df)

    return flat_samples
