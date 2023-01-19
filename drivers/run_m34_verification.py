"""
Script to check the M34 gyro age, sans training.
"""
import os
from shutil import copyfile
from os.path import join
from astropy.io import fits
from astropy.table import Table
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr

from gyrointerp.paths import DATADIR, RESULTSDIR, CACHEDIR

from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

from cdips.utils.gaiaqueries import (
    given_dr2_sourceids_get_edr3_xmatch, given_source_ids_get_gaia_data,
    given_source_ids_get_neighbor_counts
)
from gyrointerp.teff import given_dr2_BpmRp_AV_get_Teff_Curtis2020
from gyrointerp.extinctionpriors import extinction_A_V_dict
from gyrointerp.gyro_posterior import gyro_age_posterior_list

cachedir = join(DATADIR, "interim", "M34")
if not os.path.exists(cachedir): os.mkdir(cachedir)

def get_dr2_xmatch():

    fitspath = join(DATADIR, "literature", "Meibom_2011_ApJ_733_115_120_rows.fits")
    hl = fits.open(fitspath)
    df = Table(hl[1].data).to_pandas()

    dfs = []
    seqs = []

    csvpath = os.path.join(cachedir, "Meibom_2011_X_GaiaDR2.csv")

    if not os.path.exists(csvpath):
        for ix, r in df.iterrows():

            ra, dec = r['RAJ2000'], r['DEJ2000']
            seq, Vmag  = r['Seq'], r['V0mag']

            print(f"{ix}/{len(df)}")

            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
            radius = u.Quantity(3, u.arcsec)
            j = Gaia.cone_search_async(coord, radius)
            r = j.get_results()

            seq_dict = {
                26: np.int64(337172254341137280),
                39: np.int64(337140815180615424),
                52: np.int64(337165966508978688),
                55: np.int64(1635721458409799680),
                70: np.int64(337160949987261312),
                77: np.int64(337163453952398080),
                94: np.int64(337174796961953664),
                98: np.int64(1635721458409799680)
            }

            if len(r) == 1:
                dfs.append(r.to_pandas())

            elif seq in list(seq_dict.keys()):
                source_id = seq_dict[seq]
                sr = r[r['source_id'] == source_id]
                dfs.append(sr.to_pandas())

            else:
                raise NotImplementedError

            seqs.append(seq)

        seqs = np.array(seqs)
        missing_seqs = [ix+1 for ix, d in enumerate(dfs) if len(d) == 0]
        mdf_seqs = seqs[~np.in1d(seqs, np.array(missing_seqs))]

        mdf = pd.concat(dfs)
        mdf['Seq'] = mdf_seqs

        mdf.to_csv(csvpath, index=False)
        print(f"wrote {csvpath}")

    rdf = pd.read_csv(csvpath)
    mdf = rdf.merge(df, how='left', on='Seq')

    # OK -- you have all the relevant cols
    mdf = mdf.rename(
        {'source_id':'dr2_source_id', 'bp_rp': 'dr2_bp_rp'}, axis='columns'
    )
    selcols = ['dr2_source_id', 'dr2_bp_rp', 'dist', 'Seq', 'RAJ2000', 'DEJ2000',
               'Prot', 'V0mag', '__B-V_0', 'o_RVel', 'RVel', 'e_RVel', 'PRV',
               'PPM', 'Mm', 'Rot', 'JP96']
    mdf = mdf[selcols]

    return mdf


def get_dr3_xmatch(mdf):

    csvpath = os.path.join(cachedir, "Meibom_2011_X_dr2_dr3.csv")
    if os.path.exists(csvpath):
        return pd.read_csv(csvpath, dtype={'dr3_source_id':str})

    #
    # Now do the DR3 xmatch (NOTE: this is auxiliary since we're not doing binarity
    # checks)
    #
    dr2_source_ids = np.array(mdf.dr2_source_id).astype(np.int64)
    runid = "M34_dr2_to_dr3"
    dr2_x_dr3_df = given_dr2_sourceids_get_edr3_xmatch(
        dr2_source_ids, runid, overwrite=False, enforce_all_sourceids_viable=True
    )

    # Take the closest magnitude difference as the single match.
    get_dr3_xm = lambda _df: (
            _df.sort_values(by='abs_magnitude_difference').
            drop_duplicates(subset='dr3_source_id', keep='first')
    )
    s_dr3 = get_dr3_xm(dr2_x_dr3_df)
    s_dr3.dr3_source_id = s_dr3.dr3_source_id.astype(str)

    print(10*'-')
    print(s_dr3.describe())
    print(10*'-')

    ss_dr3 = s_dr3[(s_dr3.abs_magnitude_difference < 1) &
                   (s_dr3.angular_distance < 5)]

    ss_dr3 = ss_dr3.drop('source_id', axis='columns')

    smdf = mdf.merge(ss_dr3, on='dr2_source_id', how='left')
    smdf['dr3_source_id'] = smdf.dr3_source_id.astype(str)

    smdf['Teff_Curtis20'] = given_dr2_BpmRp_AV_get_Teff_Curtis2020(
        np.array(smdf.dr2_bp_rp), extinction_A_V_dict["M34"]
    )

    smdf.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")

    return smdf


def append_binarity_checks(mdf):

    dGmag = 3.25 # anything within 20x the brightness of the target
    sep_arcsec = 3 # assume seeing limited
    runid = "Meibom_2011_neighbor_count"
    count_df, _ = given_source_ids_get_neighbor_counts(
        np.array(mdf.dr3_source_id).astype(np.int64), dGmag, sep_arcsec,
        runid, overwrite=0
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

    cluster = 'M34'
    outdir = join(RESULTSDIR, 'train_verification')
    tempcsv = os.path.join(outdir, f"{cluster}.csv")
    camd_outliercsv = os.path.join(
        outdir, f"camd_outliers.csv"
    )
    if not os.path.exists(tempcsv):
        mdf.to_csv(tempcsv, index=False)
        errmsg = (
            f'Wrote {tempcsv}.  Open in glue, and manually select CAMD '
            f'outliers in M_G vs bp_rp, g_rp, bp_g and write '
            f'to {camd_outliercsv}.'
        )
        errmsg += 42*'-'
        print(42*'-')
        raise AssertionError(errmsg)

    assert os.path.exists(camd_outliercsv)

    df_camd_outlier = pd.read_csv(camd_outliercsv)

    mdf["flag_ruwe_outlier"] = mdf.ruwe > 1.2

    mdf["flag_camd_outlier"] = mdf.dr3_source_id.astype(str).isin(
        df_camd_outlier.dr3_source_id.astype(str)
    )

    mdf["flag_possible_binary"] = (
        (mdf.flag_ruwe_outlier)
        |
        (mdf.flag_camd_outlier)
        |
        (mdf.non_single_star)
        |
        (mdf.flag_nbhr_count)
        |
        (mdf.flag_bad_gaiamatch)
    )

    return mdf


def get_dr3supp(smdf):

    sel = (smdf.Mm != 'NM') & (smdf.Prot > 0) & (smdf.Teff_Curtis20 > 0)

    ssmdf = smdf[sel]

    _sel = ssmdf.dr3_source_id.astype(float) > 0
    gdf = given_source_ids_get_gaia_data(
        nparr(ssmdf[_sel].dr3_source_id).astype(np.int64), "Meibom2011_dr3",
        n_max=10000, overwrite=False, enforce_all_sourceids_viable=True,
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
    gdf['dr3_source_id'] = gdf.dr3_source_id.astype(str)
    ssmdf['dr3_source_id'] = ssmdf.dr3_source_id.astype(str)

    dr3_mdf = ssmdf.merge(gdf, on='dr3_source_id', how='inner')

    dr3_mdf = append_binarity_checks(dr3_mdf)

    return dr3_mdf


def main():

    RUN_INITIAL = 0         # to generate posteriors
    RUN_NO_BINARIES = 1     # to organie posteriors after binary selxn

    mdf = get_dr2_xmatch()
    smdf = get_dr3_xmatch(mdf)
    dr3_mdf = get_dr3supp(smdf)

    if RUN_INITIAL:
        Teffs, Prots = nparr(ssmdf.Teff_Curtis20), nparr(ssmdf.Prot)
        sel = (Teffs > 3800) & (Teffs < 6200) & (Prots > 0)
        Teffs = Teffs[sel]
        Prots = Prots[sel]

        fig, ax = plt.subplots()
        ax.scatter(Teffs, Prots, s=5, c='k', lw=0)
        ax.update({
            'xlabel': 'teff curtis20',
            'ylabel': 'prot [d]'
        })
        outdir = join(RESULTSDIR, 'train_verification')
        outpath = join(outdir, 'm34_prot_vs_teff.png')
        fig.savefig(outpath, bbox_inches='tight', dpi=300)

        cluster = "M34"
        cache_id = f"train_verification/{cluster}".replace(" ","_")

        # calculate the posterior
        age_grid = np.linspace(0, 2600, 5000) # dense grid for multiplication
        gyro_age_posterior_list(cache_id, Prots, Teffs, age_grid)

    if RUN_NO_BINARIES:

        sel = (~dr3_mdf.flag_possible_binary)

        Teffs, Prots = nparr(dr3_mdf[sel].Teff_Curtis20), nparr(dr3_mdf[sel].Prot)
        sel = (Teffs > 3800) & (Teffs < 6200) & (Prots > 0)
        Teffs = Teffs[sel]
        Prots = Prots[sel]

        fig, ax = plt.subplots()
        ax.scatter(Teffs, Prots, s=5, c='k', lw=0)
        ax.update({
            'xlabel': 'teff curtis20',
            'ylabel': 'prot [d]'
        })
        outdir = join(RESULTSDIR, 'train_verification')
        outpath = join(outdir, 'm34_prot_vs_teff_no_binaries.png')
        fig.savefig(outpath, bbox_inches='tight', dpi=300)

        cluster = "M34"
        cache_id = f"train_verification/{cluster}".replace(" ","_")

        cachedir = join(CACHEDIR, cache_id)
        dstdir = cachedir.replace("M34","M34-no-binaries")
        if not os.path.exists(dstdir): os.mkdir(dstdir)

        for Prot, Teff in zip(Prots, Teffs):

            Protstr = f"{float(Prot):.4f}"
            Teffstr = f"{float(Teff):.1f}"
            typestr = 'limitgrid'
            bounds_error = 'limit'
            paramstr = "_defaultparameters"

            fnames = [
                f"Prot{Protstr}_Teff{Teffstr}_{typestr}{paramstr}.csv",
                f"Prot{Protstr}_Teff{Teffstr}_{typestr}{paramstr}_posterior.csv",
            ]
            for fname in fnames:
                cachepath = join(cachedir, fname)
                src = cachepath
                dst = join(dstdir, os.path.basename(cachepath))
                if not os.path.exists(dst):
                    copyfile(src,dst)
                    print(f"cp->{dst}")

    # to plot, use "run_train_verificaton.py"

if __name__ == "__main__":
    main()
