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


def main():

    RUN_INITIAL = 0         # to generate posteriors
    RUN_NO_BINARIES = 1     # to organie posteriors after binary selxn

    fitspath = join(
        DATADIR, "literature",
        "Godoy-Rivera_2021_M37_M50_NGC2516_NGC2547_NGC6811_Pleiades_Praesepe.fits"
    )
    hl = fits.open(fitspath)
    d = hl[1].data
    hl.close()

    df = Table(d[d['Cluster'] == 'M37']).to_pandas()

    sel = (df.Pmemb > 0.9)
    sdf = df[sel]

    cachedir = join(DATADIR, "interim", "M37")
    if not os.path.exists(cachedir): os.mkdir(cachedir)

    sdf = sdf.rename({
        'Period':"Prot"
    }, axis='columns')

    sdf.to_csv(join(cachedir, "m37_pmembgt0pt9.csv"), index=False)

    # Pmemb > 0.98
    # CAMD manually cut in G vs BP-RP (observed)
    # A few blatant Prot vs Teff outliers omitted
    cachepath = join(
        cachedir, "m37_manual_select_cut_Pmemb0pt98_CAMD_ProtTeff.csv"
    )
    s = ''
    if os.path.exists(cachepath):
        sdf = pd.read_csv(cachepath)
        s = "_cleaned"

    #mdf = get_dr2_xmatch()
    #smdf = get_dr3_xmatch(mdf)
    #dr3_mdf = get_dr3supp(smdf)

    if RUN_INITIAL:
        Teffs, Prots = nparr(sdf.Teff), nparr(sdf.Prot)
        sel = (Teffs > 3800) & (Teffs < 6200) & (Prots > 0)
        Teffs = Teffs[sel]
        Prots = Prots[sel]

        fig, ax = plt.subplots()
        ax.scatter(Teffs, Prots, s=5, c='k', lw=0)
        ax.update({
            'xlabel': 'teff [godoyrivera21]',
            'ylabel': 'prot [d]'
        })
        outdir = join(RESULTSDIR, 'train_verification')
        outpath = join(outdir, f'm37_prot_vs_teff{s}.png')
        fig.savefig(outpath, bbox_inches='tight', dpi=300)

        cluster = "M37"
        cache_id = f"train_verification/{cluster}".replace(" ","_")

        # calculate the posterior
        age_grid = np.linspace(0, 2600, 5000) # dense grid for multiplication
        gyro_age_posterior_list(cache_id, Prots, Teffs, age_grid)

    if RUN_NO_BINARIES:

        from cdips.utils.gaiaqueries import (
            given_source_ids_get_gaia_data
        )

        source_ids = np.array(sdf.GaiaDR2).astype(np.int64)
        groupname = 'm37_hartman09'
        gdf = given_source_ids_get_gaia_data(
            source_ids, groupname, n_max=10000, overwrite=False,
            enforce_all_sourceids_viable=True, savstr='', which_columns='*',
            table_name='gaia_source', gaia_datarelease='gaiadr2',
            getdr2ruwe=True
        )

        sdf['dr2_source_id'] = np.array(sdf.GaiaDR2).astype(str)
        gdf['dr2_source_id'] = np.array(gdf.source_id).astype(str)

        smdf = sdf.merge(gdf, on='dr2_source_id', how='left')

        sel = smdf.ruwe < 1.2

        Teffs, Prots = nparr(smdf[sel].Teff), nparr(smdf[sel].Prot)
        sel = (Teffs > 3800) & (Teffs < 6200) & (Prots > 0)
        Teffs = Teffs[sel]
        Prots = Prots[sel]

        fig, ax = plt.subplots()
        ax.scatter(Teffs, Prots, s=5, c='k', lw=0)
        ax.update({
            'xlabel': 'teff galindo-guil21',
            'ylabel': 'prot [d]'
        })
        outdir = join(RESULTSDIR, 'train_verification')
        outpath = join(outdir, 'm37_prot_vs_teff_no_binaries.png')
        fig.savefig(outpath, bbox_inches='tight', dpi=300)

        cluster = "M37"
        cache_id = f"train_verification/{cluster}".replace(" ","_")

        cachedir = join(CACHEDIR, cache_id)
        dstdir = cachedir.replace("M37","M37-no-binaries")
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
