"""
Helper functions for other parts of gyro-interp.  Useful contents:

    | ``get_summary_statistics``
    | ``sample_ages_from_pdf``
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
import warnings
import os
from os.path import join
import numpy as np, pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad, IntegrationWarning

from scipy import __version__ as scipyversion
from packaging import version
scipy_ver = version.parse(scipyversion)

# Per https://docs.scipy.org/doc/scipy/release/1.12.0-notes.html
if scipy_ver >= version.parse("1.12.0"):
    from scipy.integrate import cumulative_trapezoid as integration_func
else:
    from scipy.integrate import cumtrapz as integration_func

warnings.filterwarnings(
    "ignore", category=IntegrationWarning
)

def get_summary_statistics(age_grid, age_post):
    """
    Given an age posterior probability density, ``age_post``, over a
    grid over ages, ``age_grid``, determine summary statistics for the
    posterior (its median, mean, +/-1 and 2-sigma intervals, etc).  Do
    this by interpolating over the posterior probability function.

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
        age_grid, age_post
    )


def sample_ages_from_pdf(age_grid, age_post, n_samples=1000):
    """
    Draw samples from a given age posterior probability density function (PDF)
    using quadratic interpolation.

    Args:

        age_grid (np.ndarray):
            Array of ages (in megayears) representing the grid points of the
            PDF.

        age_post (np.ndarray):
            Array of posterior probabilities corresponding to the age grid
            points.

        n_samples (int, optional):
            Number of samples to draw from the PDF. Default is 1000.

    Returns:

        np.ndarray : age_samples

            Numpy array of shape (n_samples,) containing the drawn age samples
            from the PDF.
    """
    # Normalize the posterior probability (PDF)
    age_pdf = age_post / np.trapz(age_post, age_grid)

    # Create a quadratic interpolation function for the PDF.  Go linear to
    # avoid negative values.
    pdf_interp = interp1d(age_grid, age_pdf, kind='linear')

    # Generate a fine grid of ages for sampling.  For an age_grid spanning 0 to
    # 5000 Myr, hard-coding n_grid = 1,000,000 implies a grid resolution of
    # 0.005 Myr, which is sufficiently small that the implied truncation error
    # downstream will be negligble.
    n_grid = 1000000
    age_fine_grid = np.linspace(age_grid[0], age_grid[-1], n_grid)

    # Evaluate the interpolated PDF on the fine grid
    pdf_fine = pdf_interp(age_fine_grid)

    # Normalize the interpolated PDF
    pdf_fine /= np.trapz(pdf_fine, age_fine_grid)

    # Generate random samples from the interpolated PDF
    age_samples = np.random.choice(age_fine_grid, size=n_samples,
                                   p=pdf_fine/pdf_fine.sum())

    return age_samples


def _deprecated_given_grid_post_get_summary_statistics(age_grid, age_post, N=int(1e5)):
    """
    yields results consistent within 10% of the interpolation-based
    _given_grid_post_get_summary_statistics implementation below;
    deprecated however because this approach yields quantization at
    the grid level.
    """

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


def _given_grid_post_get_summary_statistics(age_grid, age_post):

    if not np.all(np.isfinite(age_post)):
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

    age_peak = int(age_grid[np.argmax(age_post)])

    # Normalize the posterior probability (PDF)
    age_pdf = age_post / np.trapz(age_post, age_grid)

    # Calculate the cumulative distribution function (CDF)
    age_cdf = integration_func(age_pdf, age_grid, initial=0)

    # Create interpolation functions for PDF and CDF
    pdf_interp = interp1d(age_grid, age_pdf, kind='linear')
    cdf_interp = interp1d(age_grid, age_cdf, kind='linear')

    # Calculate the median age
    median_age = _find_percentile(cdf_interp, age_grid, 0.5)

    # Calculate the mean age
    mean_age = _calculate_mean(pdf_interp, age_grid)

    # Calculate the +/- 1, 2, and 3 sigma intervals
    p1sig, m1sig = _find_sigma_interval(cdf_interp, age_grid, 0.6827)
    p2sig, m2sig = _find_sigma_interval(cdf_interp, age_grid, 0.9545)
    p3sig, m3sig = _find_sigma_interval(cdf_interp, age_grid, 0.9973)

    outdict = {
        'median': np.round(median_age, 2),
        'peak': np.round(age_peak, 2),
        'mean': np.round(mean_age, 2),
        '+1sigma': np.round(p1sig, 2),
        '-1sigma': np.round(m1sig, 2),
        '+2sigma': np.round(p2sig, 2),
        '-2sigma': np.round(m2sig, 2),
        '+3sigma': np.round(p3sig, 2),
        '-3sigma': np.round(m3sig, 2),
        '+1sigmapct': np.round(p1sig / median_age, 2),
        '-1sigmapct': np.round(m1sig / median_age, 2),
    }

    return outdict


def _find_percentile(cdf_interp, age_grid, percentile):
    def objective(x):
        return cdf_interp(x) - percentile

    return _find_root(objective, age_grid[0], age_grid[-1])


def _calculate_mean(pdf_interp, age_grid):
    def integrand(x):
        return x * pdf_interp(x)

    mean, _ = quad(integrand, age_grid[0], age_grid[-1])
    return mean


def _find_sigma_interval(cdf_interp, age_grid, sigma_fraction):
    median = _find_percentile(cdf_interp, age_grid, 0.5)
    p_sigma = _find_percentile(cdf_interp, age_grid, 0.5 + sigma_fraction / 2)
    m_sigma = _find_percentile(cdf_interp, age_grid, 0.5 - sigma_fraction / 2)
    return p_sigma - median, median - m_sigma


def _find_root(func, a, b, tol=1e-6):
    fa, fb = func(a), func(b)
    assert fa * fb <= 0, "Root not bracketed"

    while abs(b - a) > tol:
        c = (a + b) / 2
        fc = func(c)
        if fc == 0:
            return c
        elif fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return (a + b) / 2


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
