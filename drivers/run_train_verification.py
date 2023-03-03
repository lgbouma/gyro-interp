"""
What ages do we get for the clusters that we trained this model on?  How
precise are the inferred cluster ages?  If someone wanted to infer the age of a
cluster using gyrointerp, are there any issues they should be aware of?

Contents:
    calc_posteriors
    plot_posteriors
    make_posterior_samples
    stack_posteriors
    run_posteriorstacker
"""

import os
from os.path import join
import matplotlib as mpl
mpl.use("agg")
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from gyrointerp.getters import _get_cluster_Prot_Teff_data
from gyrointerp.gyro_posterior import gyro_age_posterior_list
from gyrointerp.helpers import get_summary_statistics
from glob import glob
from aesthetic.plot import set_style, savefig
from gyrointerp.paths import LOCALDIR, RESULTSDIR

savdir = join(LOCALDIR, "gyrointerp", "train_verification")
if not os.path.exists(savdir): os.mkdir(savdir)

def calc_posteriors():

    #clusters = ['Pleiades', 'α Per', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
    #            'Praesepe', 'NGC-6811']
    clusters = ['NGC-6819', 'Ruprecht-147']

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

        # dense grid for multiplication
        if cluster in ['NGC-6819', 'Ruprecht-147']:
            age_grid = np.linspace(0, 5000, 5000)
        else:
            age_grid = np.linspace(0, 2700, 5000)

        # calculate the posterior
        gyro_age_posterior_list(cache_id, Prots, Teffs, age_grid)


def plot_posteriors():

    clusters = ['M34-no-binaries', 'M34']
    clusters = ['Pleiades', 'α Per', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                'Group-X', 'Praesepe', 'NGC-6811']
    clusters = ['NGC-6819', 'Ruprecht-147']
    clusters = ['M37', 'M37-no-binaries']

    summaries = []

    for cluster in clusters:

        csvdir = join(
            LOCALDIR, "gyrointerp", "train_verification",
            cluster.replace(" ","_")
        )
        csvpaths = glob(join(csvdir, "*posterior.csv"))
        assert len(csvpaths) > 0

        outdir = join(RESULTSDIR, "train_verification")
        if not os.path.exists(outdir): os.mkdir(outdir)

        # plot all stars
        plt.close("all")
        set_style('clean')
        fig, ax = plt.subplots()

        for ix, csvpath in enumerate(csvpaths):

            if not is_star_ok(csvpath):
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

                if 'M37' in cluster:
                    # avoid floating point style truncation error
                    sel = (age_grid > 500) & (age_grid < 600)
                    #print(ix, max(final_post[sel]))
                    final_post /= np.trapz(final_post, age_grid)

            zorder = ix
            ax.plot(age_grid, 1e3*t_post/np.trapz(t_post, age_grid), alpha=0.1,
                    lw=0.3, c='k', zorder=zorder)

        show_final_post = False
        if show_final_post:
            final_post = final_post/np.trapz(final_post, age_grid)
            ax.plot(age_grid, 1e3*final_post, alpha=1,
                    zorder=zorder+1, lw=0.3, c='k')

        d = get_summary_statistics(age_grid, final_post)
        print(d)
        summaries.append(d)

        xmin = 0
        xmax = 1000
        if cluster in ["NGC-6811", "Praesepe"]:
            xmax = 2000
        elif cluster in ["NGC-6819", "Ruprecht-147"]:
            xmax = 4000
        ax.update({
            'xlabel': 'Age [Myr]',
            'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
            'xlim': [xmin, xmax],
        })
        outpath = join(outdir, f'{cluster.replace(" ","_")}_verification.png')
        savefig(fig, outpath, writepdf=0)

    df = pd.DataFrame(summaries, index=clusters)
    df = df.sort_values(by='median')
    csvpath = join(outdir, "verification.csv")
    df.to_csv(csvpath, index=True)
    print(f"Wrote {csvpath}")


def is_star_ok(csvpath):

    is_fine = 1
    if 'Prot21.7500_Teff4250.1' in csvpath:
        print('Skipping Prot21.7500_Teff4250.1, it is a many-sigma '
              'outlier that yields NaN age for Praesepe.')
        is_fine = 0
    if 'Prot12.2200_Teff5048.5' in csvpath:
        print('Skipping Prot12.2200_Teff5048.5, it is a many-sigma '
              'outlier that yields too-old age for Psc-Eri.')
        is_fine = 0
    if 'Prot4.9000_Teff6187.4' in csvpath:
        print('Skipping rapid rotating hot stars in NGC-6819....')
        is_fine = 0
    if 'Prot6.3600_Teff6168.9' in csvpath:
        print('Skipping rapid rotating hot stars in NGC-6819....')
        is_fine = 0
    if 'Prot6.9079_Teff6102.6' in csvpath:
        print('Skipping rapid rotating hot stars in Ruprecht-147....')
        is_fine = 0
    return is_fine



def make_posterior_samples(N_samples=400):

    clusters = ['M34-no-binaries', 'M34', 'M37', 'M37-no-binaries']
    #clusters = ['Pleiades', 'α Per', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
    #            'Group-X', 'Praesepe', 'NGC-6811', 'NGC-6819', 'Ruprecht-147']

    for cluster in clusters:

        cstr = cluster.replace(" ", "_")
        csvdir = join(LOCALDIR, "gyrointerp", "train_verification", cstr)
        csvpaths = glob(join(csvdir, "*posterior.csv"))
        assert len(csvpaths) > 0

        outdir = join(RESULTSDIR, "train_verification_stacker")
        if not os.path.exists(outdir): os.mkdir(outdir)
        outdir = join(outdir, cstr)
        if not os.path.exists(outdir): os.mkdir(outdir)

        all_samples = []

        # draw samples from each star's posterior
        for ix, csvpath in enumerate(csvpaths):

            if not is_star_ok(csvpath):
                continue

            df = pd.read_csv(csvpath)
            t_post = np.array(df.age_post)

            star_samples = np.array(df.sample(
                n=N_samples, replace=True, weights=df.age_post
            ).age_grid)

            all_samples.append(star_samples)

        # all_samples: 2-D sample, (N_objects x N_samples)
        all_samples = np.vstack(all_samples)

        outpath = join(outdir, f"{cstr}_agepost_{N_samples}_samples.txt")
        print(f"Wrote {outpath}")
        np.savetxt(outpath, all_samples)


def run_posteriorstacker(samples_path, low=0, high=3000, nbins=11,
                         verbose=True, name='Age [Myr]'):
    """
    This function is ported from github.com/JohannesBuchner/PosteriorStacker
    Johannes Buchner (C) 2020-2021 <johannes.buchner.acad@gmx.com>

    Args:

        'samples_path', type=str, "path containing posterior samples, one object per line"

        "low", type=float, "Lower end of the distribution"

        "high", type=float, "Upper end of the distribution"

        'nbins', type=int, "Number of histogram bins"

        '--verbose', type=bool, "Show progress"

        '--name', type=str, "Parameter name (for plot)"
    """

    # `pip install posteriorstacker`
    import ultranest, ultranest.stepsampler
    __version__ = '0.6.1'

    data = np.loadtxt(samples_path)
    print(f"Loaded {samples_path}")
    basename = os.path.basename(samples_path).replace(".txt", "")
    outdir = os.path.dirname(samples_path)
    Nobj, Nsamples = data.shape
    minval = low
    maxval = high
    ndim = nbins
    viz_callback = 'auto' if verbose else None

    bins = np.linspace(minval, maxval, ndim+1)
    bins_lo = bins[:-1]
    bins_hi = bins[1:]

    binned_data = np.array([np.histogram(row, bins=bins)[0] for row in data])

    # compute KL for each object:
    prior = (1. / ndim)
    density = (binned_data + 0.1) / ((binned_data + 0.1).sum(axis=1)).reshape((-1, 1))
    KL = (density * np.log2((density / prior))).sum(axis=1)
    for i in np.argsort(KL):
        plt.plot(
            bins[:-1], binned_data[i] / binned_data[i].sum(),
            drawstyle='steps-pre',
            alpha=0.25 if KL[i] < 2 else None,
            color='gray' if KL[i] < 1 else 'k' if KL[i] < 2 else None,
            label=None if KL[i] < 2 else i
        )
    plt.xlabel(name)
    plt.ylabel('Probability density')
    plt.legend(loc='best', title='Input row', prop=dict(size=6))
    plt.savefig(join(outdir, f'{basename}_hists.pdf'), bbox_inches='tight')
    plt.close()

    print("fitting histogram model...")

    param_names = ['bin%d' % (i+1) for i in range(ndim)]

    def likelihood(params):
        """Histogram model"""
        return np.log(np.dot(binned_data, params) / Nsamples + 1e-300).sum()

    def transform_dirichlet(quantiles):
        """Histogram distribution priors"""
        # https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
        # first inverse transform sample from Gamma(alpha=1,beta=1), which is Exponential(1)
        gamma_quantiles = -np.log(quantiles)
        # dirichlet variables
        return gamma_quantiles / gamma_quantiles.sum()

    sampler = ultranest.ReactiveNestedSampler(
        param_names, likelihood, transform_dirichlet,
        log_dir=join(outdir, basename+'_out_flex%d' % ndim), resume=True
    )

    sampler.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(
        40, region_filter=False
    )

    result = sampler.run(frac_remain=0.5, viz_callback=viz_callback)
    sampler.print_results()
    sampler.plot()

    print("fitting gaussian model...")

    gparam_names = ['mean', 'std']

    def normal_pdf(x, mean, std):
        """Same as scipy.stats.norm(norm, std).pdf(x), but faster."""
        return np.exp(-0.5 * ((x - mean) / std)**2) / (std * (2 * np.pi)**0.5)

    def glikelihood(params):
        """Gaussian sample distribution"""
        mean, std = params
        return np.log(normal_pdf(data, mean, std).mean(axis=1) + 1e-300).sum()

    def gtransform(cube):
        """Gaussian sample distribution priors"""
        params = cube.copy()
        params[0] = 3 * (maxval - minval) * cube[0] + minval
        params[1] = cube[1] * (maxval - minval) * 3
        return params

    gsampler = ultranest.ReactiveNestedSampler(
        gparam_names, glikelihood, gtransform,
        log_dir=join(outdir, basename+'_out_gauss'), resume=True
    )
    gresult = gsampler.run(frac_remain=0.5, viz_callback=viz_callback)
    gsampler.print_results()


    avg_mean, avg_std = gresult['samples'].mean(axis=0)
    N_resolved = np.logical_and(
        data > avg_mean - 5 * avg_std, data < avg_mean + 5 * avg_std
    ).sum(axis=1)

    import warnings
    N_undersampled = (N_resolved < 20).sum()

    if N_undersampled > 0:
        warnings.warn("std may be over-estimated: too few samples to resolve the distribution in %d objects." % N_undersampled)

    print()
    print("Vary the number of samples to check numerical stability!")

    print("plotting results ...")
    gsampler.plot()


    plt.figure(figsize=(5,3))
    from ultranest.plot import PredictionBand

    lo_data = np.quantile(binned_data, 0.15865525393145707, axis=0)
    mid_data = np.quantile(binned_data, 0.5, axis=0)
    hi_data = np.quantile(binned_data, 0.8413447460685429, axis=0)

    x = np.linspace(minval, maxval, 400)
    band = PredictionBand(x)

    for mean, std in gresult['samples']:
        band.add(normal_pdf(x, mean, std))
    band.line(color='r', label='Gaussian model')
    band.shade(alpha=0.5, color='r')

    lo_hist = np.quantile(result['samples'], 0.15865525393145707, axis=0)
    mid_hist = np.quantile(result['samples'], 0.5, axis=0)
    hi_hist = np.quantile(result['samples'], 0.8413447460685429, axis=0)

    plt.errorbar(
        x=(bins_hi+bins_lo)/2,
        xerr=(bins_hi-bins_lo)/2,
        y=result['samples'].mean(axis=0) / (bins_hi-bins_lo),
        yerr=[
            (mid_hist - lo_hist) / (bins_hi-bins_lo),
            (hi_hist - mid_hist) / (bins_hi-bins_lo)
        ],
        marker='o', linestyle=' ', color='k',
        label='Histogram model', capsize=2)

    plt.xlabel(name)
    plt.ylabel('Probability density')
    plt.legend(loc='best')
    plt.savefig(join(outdir, f'{basename}_out.pdf'), bbox_inches='tight')
    plt.close()


def stack_posteriors(N_sample_list):

    for N_samples in N_sample_list:

        clusters = ['M34-no-binaries', 'M34', 'M37', 'M37-no-binaries']
        #clusters = ['Pleiades', 'α Per', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
        #            'Group-X', 'Praesepe', 'NGC-6811', 'NGC-6819', 'Ruprecht-147']

        for cluster in clusters:

            cstr = cluster.replace(" ", "_")

            outdir = join(RESULTSDIR, "train_verification_stacker")
            assert os.path.exists(outdir)
            outdir = join(outdir, cstr)
            assert os.path.exists(outdir)

            samples_path = join(outdir, f"{cstr}_agepost_{N_samples}_samples.txt")
            assert os.path.exists(samples_path)

            run_posteriorstacker(samples_path, low=0, high=3000, nbins=11,
                                 verbose=True, name='Age [Myr]')


def evaluate_posterior_stacker(N_sample_list):

    _dfs = []

    for N_samples in N_sample_list:

        clusters = ['M34-no-binaries', 'M34', 'M37', 'M37-no-binaries']
        #clusters = ['Pleiades', 'α Per', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
        #            'Group-X', 'Praesepe', 'NGC-6811', 'NGC-6819', 'Ruprecht-147']

        for cluster in clusters:

            cstr = cluster.replace(" ", "_")

            outdir = join(RESULTSDIR, "train_verification_stacker")
            assert os.path.exists(outdir)
            outdir = join(outdir, cstr)
            assert os.path.exists(outdir)
            outdir = join(
                outdir, f"{cstr}_agepost_{N_samples}_samples_out_gauss", "info"
            )
            assert os.path.exists(outdir)

            csvpath = join(outdir, f"post_summary.csv")
            assert os.path.exists(csvpath)

            df = pd.read_csv(csvpath)
            df['cluster'] = cluster
            df['N_samples'] = N_samples

            _dfs.append(df)

    df = pd.concat(_dfs)
    df = df.sort_values(by=['cluster','N_samples'])

    df['mean_-1sig'] = df['mean_errlo'] - df['mean_median']
    df['mean_+1sig'] = df['mean_errup'] - df['mean_median']

    selcols = ['cluster', 'N_samples', 'mean_median', 'mean_-1sig',
               'mean_+1sig']
    print(df[selcols].round(0))

    sdf = df[df.N_samples == 800]
    print(sdf[selcols].round(0))


if __name__ == "__main__":
    do_calc = 0
    do_plot = 0
    do_postsamplemaker = 0
    do_posteriorstacker = 0
    do_evaluate_poststacker = 1
    N_sample_list = [200, 400, 600, 800]

    if do_calc:
        calc_posteriors()
    if do_plot:
        plot_posteriors()
    if do_postsamplemaker:
        for N_samples in N_sample_list:
            make_posterior_samples(N_samples=N_samples)
    if do_posteriorstacker:
        stack_posteriors(N_sample_list)
    if do_evaluate_poststacker:
        evaluate_posterior_stacker(N_sample_list)
