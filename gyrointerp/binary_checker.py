"""
Contents:
    binary_checker
"""

import numpy as np

from cdips.utils.gaiaqueries import (
    given_dr2_sourceids_get_edr3_xmatch,
    given_dr3_sourceids_get_dr2_xmatch,
    given_source_ids_get_gaia_data,
    given_source_ids_get_neighbor_counts
)

def given_source_ids_return_possible_binarity(
    source_ids, gaia_datarelease,
    flag_cutoffs = {
        'rv_error': 10,
        'ruwe': 1.3,
        'dGmag': 2.5,
        'sep_arcsec': 4
    },
    runid = None,
    overwrite = False
):
    """
    This function does some simple binary checks based on the data available in
    the Gaia point source catalogs.

    It creates flags including:
        flag_non_single_star
        flag_rv_error
        flag_ruwe
        flag_nbhr_count
        flag_possible_binary

    Something that a user might wish consider is also whether their star(s) of
    interest is a photometric binary.  This is generally best assessed using a
    comparison population.  Ideally this would be a cluster, but sometimes the
    field is the best thing available.  In such cases, comparing against a ZAMS
    locus (e.g., Pecaut & Mamajek's table), or comparing against a local volume
    of stars with similar color is recommended.  (And not implemented here!)

    ----------

    Required Args:
        source_ids (np.ndarray): a numpy array containing string-value Gaia
        source_ids.  For example, np.array(["446488105559389568"]).

        gaia_datarelease (str): Gaia data release corresponding to
        `source_ids`, must be one of "gaiadr2", "gaiadr3", or "gaiaedr3".

    Kwargs:
        flag_cutoffs (dict): containing key/value pairs for the binarity
        indicator cutoffs, described below.

        runid (None or str): string used for cacheing.  If given, results will
        be cached to a file at `~/.gaia_cache/{runid}*.*`.  Otherwise, the
        cached file will be based on the first source_id in source_ids.

        overwrite (bool): if True, forces re-running and overwrite of Gaia
        archive queries.

    Returns:
        Two dataframes: `target_df` and `nbhr_df`.

        The first is a N-row dataframe containing standard Gaia data for each
        requested source, and a set of flags that can be indicative of
        binarity:

            flag_non_single_star: Non-zero `non_single_star` bitflag from
                Gaia DR3.  Bit 1 in `non_single_star` means the Gaia pipeline
                identified it as an astrometric binary.  Bit 2 means it
                identified it as a spectroscopic binary.  Bit 3 means it
                identified it as an eclipsing binary.
                (https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)

            flag_rv_error: gaia radial_velocity_error is above
                `flag_cutoffs['rv_error']` km/s.  This means the scatter in the RV
                time-series is high.

            flag_ruwe: RUWE is above `flag_cutoffs['ruwe']`.  This can indicate
                true astrometric motion, or that there are multiple
                marginally-resolved point sources that induce extra astrometric
                scatter along each different scan direction.

            nbhr_count: count of stars within a brightness difference of
                `flag_cutoffs['dGmag']` G-band magnitudes of the target star,
                within a distance of `flag_cutoffs['sep_arcsec']`.  Note that these
                are just visually coincident stars, and needn't be binary
                companions!  Checking the parallaxes and proper motions is a more
                convincing way to demonstrate whether the stars are co-spatial
                and/or co-moving.

            flag_nbhr_count: if nbhr_count >= 1

            flag_possible_binary: bitwise-OR of the above three flags.

        The second, `nbhr_df`, contains Gaia info for the neighbors, including
        their source_id's (listed as the "source_id_2" column), distances, and
        G-band magnitude differences.
    """

    assert isinstance(source_ids, np.ndarray)
    assert isinstance(source_ids[0], str)
    assert gaia_datarelease in ['gaiadr2', 'gaiadr3', 'gaiaedr3']

    if runid is None:
        runid_str = source_ids[0]
    else:
        runid_str = runid+"_"+source_ids[0]

    #
    # First, if needed, get Gaia DR3 source_ids.
    #
    if gaia_datarelease == 'gaiadr2':
        dr2_source_ids = source_ids.astype(np.int64)
        _runid = f"{runid_str}_dr2dr3_xmatch"
        dr2_x_edr3_df = given_dr2_sourceids_get_edr3_xmatch(
            dr2_source_ids, _runid, overwrite=False,
            enforce_all_sourceids_viable=True
        )

        # If multiple matches, take the closest (proper motion and
        # epoch-corrected) angular distance as THE single match.
        if len(dr2_x_edr3_df) > 1:

            get_edr3_xm = lambda _df: (
                _df.sort_values(by='angular_distance').
                drop_duplicates(subset='dr2_source_id', keep='first')
            )
            s_edr3 = get_edr3_xm(dr2_x_edr3_df)

            dr3_source_ids = np.array(s_edr3.dr3_source_id).astype(np.int64)

    else:
        dr3_source_ids = source_ids.astype(np.int64)

    #
    # Get the Gaia data.
    #
    _runid = f"{runid_str}_dr3_data"
    gdf = given_source_ids_get_gaia_data(
        dr3_source_ids, _runid, n_max=2*len(dr3_source_ids),
        overwrite=overwrite, enforce_all_sourceids_viable=True,
        which_columns='*', gaia_datarelease='gaiadr3'
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

    dGmag = flag_cutoffs['dGmag']
    sep_arcsec = flag_cutoffs['sep_arcsec']
    runid = f"{runid_str}_neighbor_count"
    count_df, nbhr_df = given_source_ids_get_neighbor_counts(
        np.array(gdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=overwrite
    )

    assert np.all(
        count_df.source_id.astype(str) == gdf.dr3_source_id.astype(str)
    )

    gdf['nbhr_count'] = count_df['nbhr_count']

    #
    # Define flags
    #
    gdf['flag_non_single_star'] = (
        gdf['non_single_star'] > 0
    )

    gdf['flag_nbhr_count'] = (
        gdf['nbhr_count'] >= 1
    )

    gdf["flag_ruwe"] = gdf.ruwe > flag_cutoffs['ruwe']

    gdf["flag_rv_error"] = gdf.radial_velocity_error > flag_cutoffs['rv_error']

    ## NOTE: FIXME TODO query local neighborhood
    #gdf["flag_camd_outlier"] = gdf.dr3_source_id.astype(str).isin(
    #    df_camd_outlier.dr3_source_id.astype(str)
    #)

    # Benchmark periods:
    # Include:
    # * Clean quality flag only
    # Exclude:
    # * All possible signs of binarity
    # * Any missing EDR3 measurements.

    gdf["flag_possible_binary"] = (
        (gdf.flag_ruwe)
        |
        #(gdf.flag_camd_outlier)
        #|
        (gdf.flag_rv_error)
        |
        (gdf.flag_non_single_star)
        |
        (gdf.flag_nbhr_count)
    )

    return gdf, nbhr_df
