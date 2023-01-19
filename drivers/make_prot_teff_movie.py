"""
Draw N points from a cluster.  How does their evolution compare with data?
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

outdir = os.path.join(RESULTSDIR, "movie_prot_teff")
if not os.path.exists(outdir): os.mkdir(outdir)

def do_model_movie(
    Teff_obs, ages,
    movie_durn=120*u.second, # movie run-time (frame-rate is scaled accordingly)
):

    # Do the plot
    plt.close("all")
    set_style("science")
    fig, ax = plt.subplots(figsize=(3,2.5), dpi=400)

    # get initial data to plot for first frame
    _x = Teff_obs
    _y = get_Prot_model(ages[0], Teff_obs)

    numframes = len(ages)
    numpoints = len(Teff_obs)

    scat = ax.scatter(_x, _y, s=2, c='k', lw=0)
    txt = ax.text(0.05, 0.95, f"{ages[0]} Myr", transform=ax.transAxes,
                  ha='left', va='top', color='k')

    ax.set_ylabel("Rotation Period [days]")
    ax.set_xlim([7100, 2900])
    ax.set_xticks([7000, 6000, 5000, 4000, 3000])
    minor_xticks = np.arange(3000, 7100, 100)[::-1]
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_ylim([-0.5, 21])
    ax.set_yticks([0, 5, 10, 15, 20])

    ax.set_xlabel("Effective Temperature [K]")
    fig.tight_layout()

    frame_interval = (movie_durn / numframes).to(u.millisecond).value
    print(f'Doing a movie {movie_durn} long.')
    print(f'Frame interval is {frame_interval} milliseconds.')

    ani = animation.FuncAnimation(
        fig, # figure object which gets needed events
        update_plot, # function to call at each frame
        frames=range(numframes),
        interval=frame_interval,  # delay btwn frames (msec)
        fargs=(Teff_obs, ages, scat, txt)
    )

    outpath = join(outdir, f'model_prot_vs_teff.mp4')
    ani.save(outpath)


def update_plot(i, x, ages, scat, txt):

    scat.set_offsets(np.c_[x, get_Prot_model(ages[i], x)])

    txt.set_text(f'{int(ages[i])} Myr')

    return scat, txt,


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

    Teffs = np.linspace(3800, 6200, 200)
    ages = np.arange(80, 2000+1, 1)

    do_model_movie(Teffs, ages)


if __name__ == "__main__":
    main()
