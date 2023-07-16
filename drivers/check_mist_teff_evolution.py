"""
Plot Teff vs time for stars of different masses, using the data from Choi+2016
(MIST).

Generated using the web interpolator,
https://waps.cfa.harvard.edu/MIST/interp_tracks.html,
with 0.2-1.2 Msun, spaced by 0.1 Msun.
"""
from read_mist_model import ISO, EEP
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from gyrointerp.paths import RESULTSDIR
import os
from aesthetic.plot import set_style, savefig
from scipy.interpolate import make_interp_spline, BSpline
from glob import glob
from matplotlib import cm

# solar metallicity default isochrone from
# https://waps.cfa.harvard.edu/MIST/model_grids.html
masses = np.round(np.arange(0.2, 1.2+0.1, 0.1),1).astype(str)
SECRETDATADIR = '../secret_data/'
eep_paths = sorted(glob(os.path.join( SECRETDATADIR, 'mist', "*.track.eep")))

dfd = {}

for mass, eep_path in zip(masses, eep_paths):

    eep = EEP(eep_path)

    params = ['star_age', 'log_Teff', 'log_L', 'log_R', 'center_h1', 'mass_conv_core',
              'phase']

    df = pd.DataFrame({})
    for param in params:
        df[param] = eep.eeps[param]

    df['Teff'] = 10**df.log_Teff
    df['R'] = 10**df.log_R

    dfd[mass] = df

params = ['Teff', 'log_L', 'R', 'center_h1', 'mass_conv_core', 'phase']

labels = [
    'Effective temperature [K]',
    'Luminosity [L$_\odot$]',
    'Stellar radius [R$_\odot$]',
    'center_h1', 'mass_conv_core', 'phase'
]
dlabels = [
    '(Teff(t) - Teff_final)/Teff_final',
    'dL/L_final',
    'dRstar/Rstar_final',
    'dcenter_h1', 'dmass_conv_core', 'dphase'
]

yticks = [
    [3000, 4000, 5000, 6000, 8000],
    None,
    [0.1, 0.2, 0.4, 0.6, 1.0, 2.0],
    None,None,None
]
yticklabels = [
    [3000, 4000, 5000, 6000, 8000],
    None,
    [0.1, 0.2, 0.4, 0.6, 1.0, 2.0],
    None,None,None
]

outdir = os.path.join(RESULTSDIR, "mist_param_evolution")
if not os.path.exists(outdir): os.mkdir(outdir)

for differential in [1,0]:
    for youngerages in [1,0]:
        _labels = dlabels if differential else labels
        for param, l, yt, ytl in zip(params, _labels, yticks, yticklabels):

            d = '' if not differential else "diff-"
            ya = "_youngerages" if youngerages else ""
            outpath = os.path.join(outdir, f"{d}{param}_vs_age{ya}.png")

            plt.close("all")
            set_style("clean")
            fig, ax = plt.subplots()

            N_colors = len(masses)
            cmap = cm.cividis(np.linspace(0,1,N_colors))

            for ix, mass in enumerate(masses):

                df = dfd[mass]
                MAX_AGE = 2.7e9
                if youngerages:
                    sel = (df.star_age > 10e6) & (df.star_age < MAX_AGE)
                else:
                    sel = (df.star_age > 80e6) & (df.star_age < MAX_AGE)
                df = df[sel].reset_index()

                color = cmap[ix]

                if differential:
                    ix = np.argmax(df.star_age)
                    yval = (df[param] - df.loc[ix, param])/(df.loc[ix, param])
                else:
                    yval = df[param]
                dy = 0 if not differential else df.loc[ix, param]
                ax.scatter(
                    df['star_age'], yval, marker='o', c=color, s=1
                )

                xnew = np.linspace(np.log10(df.star_age.min()),
                                   np.log10(df.star_age.max()), 1000)

                # literally just linearly interpolating.
                spl = make_interp_spline(
                    np.log10(df['star_age']), np.log10(yval), k=1
                )
                smooth = spl(xnew)

                ax.plot(
                    10**xnew, 10**smooth, lw=0.5, c=color, zorder=-1, label=mass
                )

            xlim = [10e6, MAX_AGE] if youngerages else [80e6, MAX_AGE]
            ax.update({
                'xlim': xlim,
                'xlabel': 'Time [yr]',
                'xscale': 'log',
                'ylabel': l,
            })
            if not differential:
                ax.set_yscale('log')
            ax.legend(loc='best', fontsize='xx-small')
            if yt is not None and not differential:
                    ax.set_yticklabels(yt)
                    ax.set_yticks(yt)
            ax.grid(True, which='both', ls='--', lw=0.3, zorder=-2)
            savefig(fig, outpath, writepdf=1, dpi=400)
