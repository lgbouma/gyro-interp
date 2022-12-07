"""
Reusable functions.

    given_grid_post_get_summary_statistics
    left_merge
    prepend_colstr
    given_dr2_get_dr3_dataframes
"""
import os
import numpy as np, pandas as pd

def given_grid_post_get_summary_statistics(age_grid, age_post, N=int(1e5)):
    """
    Given an age posterior over a grid, determine summary statistics (peak
    location, +/-sigma intervals, etc).  Do this by sampling `N` times, with
    replacement, from the posterior, weighting by the probability.

    Returns:
        dictionary containing the summary statistics.
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
        'median': np.round(pct_50,3),
        'peak': np.round(age_peak,3),
        'mean': np.round(np.nanmean(sample_df.age),3),
        '+1sigma': np.round(p1sig,3),
        '-1sigma': np.round(m1sig,3),
        '+2sigma': np.round(p2sig,3),
        '-2sigma': np.round(m2sig,3),
        '+3sigma': np.round(p3sig,3),
        '-3sigma': np.round(m3sig,3),
        '+1sigmapct': np.round(p1sig/pct_50,3),
        '-1sigmapct': np.round(m1sig/pct_50,3),
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
    """
    dr2_source_ids: np.ndarray of np.int64 Gaia DR2 source identifiers.
    runid_dr2: arbitrary string to identify the DR2->DR3 xmatch query
    runid_dr3: arbitrary (different) string to identify the DR3 query
    """

    # pip install cdips
    from cdips.utils.gaiaqueries import (
        given_dr2_sourceids_get_edr3_xmatch, given_source_ids_get_gaia_data
    )

    print(42*'-')
    print(runid_dr2)

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
    print(10*'-')
    print(s_dr3.describe())
    print(10*'-')
    if len(s_dr3) != len(np.unique(dr2_source_ids)):
        print('Got bad dr2<->dr3 match')
        print(len(s_dr3), len(np.unique(dr2_source_ids)))
        import IPython; IPython.embed()
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
