"""
Plot M_star, Teff, L_star, and R_star of stars that have just arrived on the
ZAMS, using the data from Choi+2016 (MIST).

Writes a cached CSV containing the output for easy interpolation in
gyrointerp.models.teff_zams
"""
from read_mist_model import ISO
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from gyrointerp.paths import DATADIR, RESULTSDIR
import os
from aesthetic.plot import set_style, savefig
from scipy.interpolate import make_interp_spline, BSpline

# solar metallicity default isochrone from
# https://waps.cfa.harvard.edu/MIST/model_grids.html
iso_path = os.path.join(
    DATADIR, 'mist', "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_full.iso"
)

iso = ISO(iso_path)

# At any given age, what is the minimum mass / teff for stars that have just
# reached the ZAMS?  The default MIST grid are uniformly spaced in "EEP", which
# is EXACTLY what we want for this experiment.
# Past 9.2, goes to lower mass stars.
ages = np.array(iso.ages)
ages = ages[(ages >= 7) & (ages <= 9.2)]

sel_rows = []

for ix, age in enumerate(ages):
    if ix % 10 == 0:
        print(f"{ix}/{len(ages)}: logt {age}...")
    age_ind = iso.age_index(age)
    params = ['star_mass', 'initial_mass', 'log_Teff', 'log_L', 'log_R', 'EEP',
              'phase', 'mass_conv_core', 'log_center_T', 'log_center_Rho',
              'surface_he3', 'pp']
    df = pd.DataFrame({})
    for param in params:
        df[param] = iso.isos[age_ind][param]

    # per https://waps.cfa.harvard.edu/MIST/README_tables.pdf
    # EEP 202 is stars that have just reached the ZAMS
    sel = (df.EEP == 202)

    sdf = df[sel]
    sdf['age'] = age
    sel_rows.append(sdf)

df = pd.concat(sel_rows)

df['Teff'] = 10**df.log_Teff
df['L'] = 10**df.log_L
df['R'] = 10**df.log_R

csvpath = os.path.join(
    DATADIR, 'literature',
    'Choi_2016_MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_full_arrival_times.csv'
)
df.to_csv(csvpath, index=False)
print(f"Wrote {csvpath}")

params = ['star_mass', 'Teff', 'L', 'R',
          'mass_conv_core', 'log_center_T', 'log_center_Rho', 'surface_he3',
          'pp']
labels = [
    'Stellar mass [M$_\odot$]',
    'Effective temperature [K]',
    'Luminosity [L$_\odot$]',
    'Stellar radius [R$_\odot$]',
      'mass_conv_core', 'log_center_T', 'log_center_Rho', 'surface_he3',
      'pp'
]
yticks = [
    [0.1, 0.2, 0.4, 0.6, 1.0, 2.0],
    [3000, 4000, 5000, 6000, 8000],
    None,
    None,
    None,None,None,None,None
]
yticklabels = [
    [0.1, 0.2, 0.4, 0.6, 1.0, 2.0],
    [3000, 4000, 5000, 6000, 8000],
    None,
    None,
    None,None,None,None,None
]


outdir = os.path.join(RESULTSDIR, "mist_zams_arrival")
if not os.path.exists(outdir): os.mkdir(outdir)

for param, l, yt, ytl in zip(params, labels, yticks, yticklabels):

    outpath = os.path.join(outdir, f"{param}_vs_age.png")

    plt.close("all")
    set_style("clean")
    fig, ax = plt.subplots()
    ax.scatter(
        10**df['age'], df[param], marker='o', c='k', s=2
    )

    xnew = np.linspace(7, 9.2, 1000)

    # literally just linearly interpolating.
    spl = make_interp_spline(df['age'], np.log10(df[param]), k=1)  # type: BSpline
    smooth = spl(xnew)

    ax.plot(
        10**xnew, 10**smooth, lw=0.5, c='k', zorder=-1
    )

    ax.update({
        'xlabel': 'Time [yr]',
        'xscale': 'log',
        'ylabel': l,
        'yscale': 'log',
    })
    if yt is not None:
        ax.set_yticks(yt)
        ax.set_yticklabels(yt)
    ax.grid(True, which='both', ls='--', lw=0.3, zorder=-2)
    savefig(fig, outpath, writepdf=1, dpi=400)
