"""
Plot Prot vs Teff at fixed cluster ages to compare against the models.
"""

import os
from os.path import join
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp.getters import _get_cluster_Prot_Teff_data
from gyrointerp.models import slow_sequence, slow_sequence_residual
from gyrointerp.age_scale import agedict

from aesthetic.plot import set_style, savefig
from gyrointerp.paths import RESULTSDIR

outdir = os.path.join(RESULTSDIR, "movie_prot_teff")
if not os.path.exists(outdir): os.mkdir(outdir)

def do_cluster_plot(Teff_obs, Prot_obs, age, cluster):

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

    plt.close("all")
    set_style("science")
    fig, axs = plt.subplots(figsize=(6,2.5), ncols=2)

    ax = axs[0]
    ax.scatter(Teff_obs, Prot_obs, s=2, c='k', lw=0)
    ax.text(0.03, 0.97, "Data", transform=ax.transAxes,
            ha='left', va='top', color='k')

    ax = axs[1]
    ax.scatter(Teff_obs, Prot_mod, s=2, c='k', lw=0)
    ax.text(0.05, 0.95, "Model", transform=ax.transAxes,
            ha='left', va='top', color='k')

    for ix, ax in enumerate(axs):

        if ix == 0:
            ax.set_ylabel("Rotation Period [days]")

        ax.set_xlim([7100, 2900])
        ax.set_xticks([7000, 6000, 5000, 4000, 3000])
        minor_xticks = np.arange(3000, 7100, 100)[::-1]
        ax.set_xticks(minor_xticks, minor=True)

        ax.set_ylim([-0.5, 16])
        ax.set_yticks([0, 5, 10, 15])

    fig.text(0.5, 0.97, f"{cluster} ({age} Myr)", ha='center', va='top')
    fig.text(0.5, -0.03, "Effective Temperature [K]", ha='center')

    a = str(age).zfill(6)
    c = cluster.replace(" ","_")
    outpath = join(outdir, f'{a}_prot_vs_teff_{c}.png')

    savefig(fig, outpath, dpi=400, writepdf=0)


def main():

    #
    # Cluster plots
    #
    np.random.seed(42)

    d = _get_cluster_Prot_Teff_data()

    # Draw literally the cluster Teff distribution
    clusters = ['Pleiades', 'α Per', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
                'Praesepe', 'NGC-6811']

    for cluster in clusters:

        df = d[cluster][0]
        sel = df.flag_benchmark_period

        Teffs = np.array(df[sel].Teff_Curtis20)
        Prots = np.array(df[sel].Prot)

        sel = (Teffs > 3800) & (Teffs < 6200) & (Prots > 0)
        Teff_obs = Teffs[sel]
        Prot_obs = Prots[sel]

        age = agedict['default'][cluster]

        do_cluster_plot(Teff_obs, Prot_obs, age, cluster)


if __name__ == "__main__":
    main()
