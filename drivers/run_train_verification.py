"""
What ages do we get for the clusters that we trained this model on?  How
precise are the inferred cluster ages?  If someone wanted to infer the age of a
cluster using gyrointerp, are there any issues they should be aware of?
"""

import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from gyrointerp.getters import _get_cluster_Prot_Teff_data
from gyrointerp.gyro_posterior import gyro_age_posterior_list
from gyrointerp.helpers import given_grid_post_get_summary_statistics
from glob import glob
from aesthetic.plot import set_style, savefig
from gyrointerp.paths import LOCALDIR, RESULTSDIR

savdir = os.path.join(LOCALDIR, "gyrointerp", "train_verification")
if not os.path.exists(savdir): os.mkdir(savdir)

def calc_posteriors():

    clusters = ['Pleiades', 'α Per', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
                'Praesepe', 'NGC-6811']

    d = _get_cluster_Prot_Teff_data()

    for cluster in clusters:

        print(f"{cluster}...")

        df = d[cluster][0]
        sel = df.flag_benchmark_period

        Teffs = np.array(df[sel].Teff_Curtis20)
        Prots = np.array(df[sel].Prot)

        sel = (Teffs > 3800) & (Teffs < 6200) & (Prots > 0)
        Teffs = Teffs[sel]
        Prots = Prots[sel]

        cache_id = f"train_verification/{cluster}".replace(" ","_")

        # calculate the posterior
        age_grid = np.linspace(0, 2600, 5000) # dense grid for multiplication
        gyro_age_posterior_list(cache_id, Prots, Teffs, age_grid)


def plot_posteriors():

    clusters = ['Pleiades', 'α Per', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                'Group-X', 'Praesepe', 'NGC-6811']

    summaries = []

    for cluster in clusters:

        csvdir = os.path.join(
            LOCALDIR, "gyrointerp", "train_verification",
            cluster.replace(" ","_")
        )
        csvpaths = glob(os.path.join(csvdir, "*posterior.csv"))
        assert len(csvpaths) > 0

        outdir = os.path.join(RESULTSDIR, "train_verification")
        if not os.path.exists(outdir): os.mkdir(outdir)

        # plot all stars
        plt.close("all")
        set_style('clean')
        fig, ax = plt.subplots()

        for ix, csvpath in enumerate(csvpaths):

            if 'Prot21.7500_Teff4250.1' in csvpath:
                print('Skipping Prot21.7500_Teff4250.1, it is a many-sigma '
                      'outlier that yields NaN age for Praesepe.')
                continue

            # initialize the posterior
            if ix == 0:
                df = pd.read_csv(csvpath)
                t_post = np.array(df.age_post)
                final_post = t_post*1.
                age_grid = np.array(df.age_grid)

            # multiply elementwise -- no logs needed
            else:
                df = pd.read_csv(csvpath)
                t_post = np.array(df.age_post)
                final_post *= t_post

            zorder = ix
            ax.plot(age_grid, 1e3*t_post/np.trapz(t_post, age_grid), alpha=0.1,
                    lw=0.3, c='k', zorder=zorder)

        show_final_post = False
        if show_final_post:
            final_post = final_post/np.trapz(final_post, age_grid)
            ax.plot(age_grid, 1e3*final_post, alpha=1,
                    zorder=zorder+1, lw=0.3, c='k')

        d = given_grid_post_get_summary_statistics(age_grid, final_post)
        print(d)
        summaries.append(d)

        xmax = 1000 if cluster not in ["NGC-6811", "Praesepe"] else 2000
        ax.update({
            'xlabel': 'Age [Myr]',
            'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
            'xlim': [0, xmax],
        })
        outpath = os.path.join(outdir, f'{cluster.replace(" ","_")}_verification.png')
        savefig(fig, outpath, writepdf=0)

    df = pd.DataFrame(summaries, index=clusters)
    df = df.sort_values(by='median')
    csvpath = os.path.join(outdir, "verification.csv")
    df.to_csv(csvpath, index=True)
    print(f"Wrote {csvpath}")

    selcols = ['median','+1sigma','-1sigma']
    tdf = df.round(0).astype(int)
    tdf = tdf[selcols]
    texpath = os.path.join(outdir, "verification.tex")
    tdf.to_latex(texpath)
    print(f"Wrote {texpath}")


if __name__ == "__main__":
    do_calc = 0
    do_plot = 1
    if do_calc:
        calc_posteriors()
    if do_plot:
        plot_posteriors()