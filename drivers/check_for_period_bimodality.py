"""
If you draw 10,000 stars, from a uniform distribution in age between 0 and 4
gyr, do you reproduce the observed "period bimodality" that is seen in the
Kepler and K2 data?   (especially the K and M dwarfs...)
"""

import os
from os.path import join
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.animation as animation

from gyrointerp.getters import _get_cluster_Prot_Teff_data
from gyrointerp.models import slow_sequence, slow_sequence_residual
from gyrointerp.age_scale import agedict

from aesthetic.plot import set_style, savefig
from gyrointerp.paths import RESULTSDIR

from astropy import units as u

outdir = os.path.join(RESULTSDIR, "period_bimodality")
if not os.path.exists(outdir): os.mkdir(outdir)

def get_Prot_model(age, Teff_obs):

    # Get model Prots
    age = int(age)

    Prot_mu = slow_sequence(Teff_obs, age, verbose=False)

    y_grid = np.linspace(-14, 6, 1000)
    Prot_resid = slow_sequence_residual(age, y_grid=y_grid, teff_grid=Teff_obs)

    Prot_mod = []

    for ix, _ in enumerate(Teff_obs):

        dProt = np.random.choice(
            y_grid, size=1, replace=True,
            p=Prot_resid[:,ix]/np.sum(Prot_resid[:,ix])
        )

        Prot_mod.append(Prot_mu[ix] + dProt)

    Prot_mod = np.array(Prot_mod)

    return Prot_mod

def main():

    SEED = 42
    NSTARS = int(2e4)
    outcsv = join(outdir, f'teff_age_prot_seed{SEED}_nstar{NSTARS}.csv')

    if os.path.exists(outcsv):
        print(f"Found {outcsv}, loading")
        df = pd.read_csv(outcsv)

    else:
        np.random.seed(SEED)
        teffs = np.random.uniform(low=3800, high=6200, size=NSTARS)
        ages = np.random.uniform(low=0, high=4000, size=NSTARS)

        prot_mods = []
        ix = 0
        for teff, age in zip(teffs, ages):
            print(ix)
            prot_mod = get_Prot_model(age, np.array([teff]))
            prot_mods.append(float(prot_mod))
            ix += 1

        df = pd.DataFrame({
            'teff': teffs,
            'age': ages,
            'prot_mod': prot_mods
        })
        df.to_csv(outcsv, index=False)

    for logy in [0,1]:
        plt.close("all")
        fig, ax = plt.subplots()
        set_style("clean")

        ax.scatter(df.teff, df.prot_mod, s=1, c='k', marker='o', zorder=1, linewidths=0)

        ax.set_xlim([6200,3800])

        ax.set_xlabel("Teff [K]")
        ax.set_ylabel('Prot model [d]')
        s = ''
        if logy:
            ax.set_yscale('log')
            ax.set_ylim([1,30])
            s += '_logy'

        outpath = join(outdir, f'teff_age_prot_seed{SEED}_nstar{NSTARS}{s}.png')
        savefig(fig, outpath, writepdf=0)



if __name__ == "__main__":
    main()
