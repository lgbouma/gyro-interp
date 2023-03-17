"""
Write the "data behind the figure" table for Figure 1.  Include dr2_source_id,
dr3_source_id, Prot, Teff, DR2 BP-RP, and auxiliary flags.
"""

import os
import pandas as pd
from gyrointerp.getters import _get_cluster_Prot_Teff_data
d = _get_cluster_Prot_Teff_data()

clusters = {
    'α Per': [
        "2022arXiv221109822B"
    ],
    'Pleiades': [
        "2016AJ....152..114R"
    ],
    'Blanco-1': [
        "2020MNRAS.492.1008G"
    ],
    'Psc-Eri': [
        "2019AJ....158...77C"
    ],
    'NGC-3532': [
        "2021A&A...652A..60F"
    ],
    'Group-X': [
        "2022A&A...657L...3M"
    ],
    'Praesepe': [
        "2021ApJ...921..167R"
    ],
    'NGC-6811': [
        "2019ApJ...879...49C"
    ],
    'NGC-6819': [
        "2015Natur.517..589M"
    ],
    'Ruprecht-147': [
        "2020ApJ...904..140C"
    ]
}

selcols = [
    'dr2_source_id', 'dr3_source_id', "Prot", "dr2_bp_rp", "Teff_Curtis20",
    "phot_g_mean_mag",
    "ruwe",
    "flag_camd_outlier", "flag_ruwe_outlier", "flag_rverror_outlier",
    "flag_nbhr_count", "non_single_star", "flag_possible_binary",
    "flag_pass_author_quality", "flag_benchmark_period"
]

dfs = []

for c, bibcode in clusters.items():

    _df = d[c][0]

    if c == 'α Per':
        # Boyle+22 included a lot of faint candidate cluster members that were
        # too faint to yield TESS rotation periods.  omit them, along with any
        # candidate alpha-Per members that did not yield rotation periods.
        sel = ~pd.isnull(_df.Prot)
        _df = _df[sel]

    _df['dr2_source_id'] = _df.dr2_source_id.astype(str)
    _df['dr3_source_id'] = _df.dr3_source_id.astype(str)

    flagcols = [c for c in selcols if 'flag_' in c]
    for flagcol in flagcols:
        if flagcol not in _df:
            print(f"{flagcol} not in {c}")
            _df[flagcol] = False

    _sdf = _df[selcols]
    _sdf['Prot_provenance'] = bibcode[0]
    _sdf['cluster'] = c

    dfs.append(_sdf)

mdf = pd.concat(dfs)

mdf = mdf.rename({
    'phot_g_mean_mag': 'dr3_phot_g_mean_mag',
    'ruwe': 'dr3_ruwe',
    'non_single_star': 'dr3_non_single_star'
}, axis='columns')

mdf['Prot'] = mdf['Prot'].round(decimals=4)
mdf['Teff_Curtis20'] = mdf.Teff_Curtis20.round(decimals=1)
mdf['dr2_bp_rp'] = mdf.dr2_bp_rp.round(decimals=3)
mdf['dr3_phot_g_mean_mag'] = mdf.dr3_phot_g_mean_mag.round(decimals=3)
mdf['dr3_ruwe'] = mdf.dr3_ruwe.round(decimals=3)

mdf = mdf[~pd.isnull(mdf.dr3_non_single_star)] # drop 5 Boyle&Bouma NaN matches
mdf['dr3_non_single_star'] = mdf.dr3_non_single_star.astype(int)

outdir = "/Users/luke/Dropbox/proj/young-KOIs/paper"
outpath = os.path.join(outdir, "tab_rotation_periods.csv")
mdf.to_csv(outpath, index=False)
print(f"Wrote {outpath}")
