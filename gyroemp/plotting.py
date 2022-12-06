"""
Catch-all file for plotting scripts.  Contents:

    plot_prot_vs_teff
    plot_prot_vs_teff_residual
    plot_sub_praesepe_selection_cut
    plot_slow_sequence_residual
    plot_age_posterior
    plot_age_posteriors
    plot_cdf_fast_slow_ratio
    plot_data_vs_model_prot
    plot_fit_gyro_model
    plot_empirical_limits_of_gyrochronology

Helpers:
    _given_ax_append_spectral_types

    Sub-plot makers, to prevent code duplication:
        _plot_slow_sequence_residual
        _plot_prot_vs_teff_residual
"""
import os, pickle
from glob import glob
from os.path import join
from itertools import product
import numpy as np, pandas as pd, matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from numpy import array as nparr

from gyroemp.paths import DATADIR, RESULTSDIR, LOCALDIR
from gyroemp.models import (
    reference_cluster_slow_sequence, slow_sequence, slow_sequence_residual
)
from gyroemp.getters import _get_cluster_Prot_Teff_data
from gyroemp.helpers import given_grid_post_get_summary_statistics
from gyroemp.gyro_posterior import gyro_age_posterior
from gyroemp.age_scale import agedict

from scipy.interpolate import interp1d

# pip install aesthetic
from aesthetic.plot import set_style, savefig

###########
# helpers #
###########
def _given_ax_append_spectral_types(
    ax,
    _sptypes=['F2V','F5V','G2V','K0V','K5V','M0V','M3V']):
    # Append SpTypes (ignoring reddening)
    from cdips.utils.mamajek import get_SpType_Teff_correspondence

    tax = ax.twiny()
    xlim = ax.get_xlim()
    getter = get_SpType_Teff_correspondence
    sptypes, xtickvals = getter(
        _sptypes
    )
    print(sptypes)
    print(xtickvals)

    xvals = np.linspace(min(xlim), max(xlim), 100)
    tax.plot(xvals, np.ones_like(xvals), c='k', lw=0) # hidden, but fixes axis.
    tax.set_xlim(xlim)
    ax.set_xlim(xlim)

    tax.set_xticks(xtickvals)
    tax.set_xticklabels(sptypes, fontsize='medium')

    tax.xaxis.set_ticks_position('top')
    tax.tick_params(axis='x', which='minor', top=False)
    tax.get_yaxis().set_tick_params(which='both', direction='in')


############
# plotters #
############
def plot_prot_vs_teff(outdir, reference_clusters, show_binaries=0,
                      model_ids=None, poly_order=7,
                      slow_seq_ages=None):
    """
    reference_clusters: list containing any of
        ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X', 'Praesepe',
        'NGC-6811', 'NGC-6819', 'Ruprecht-147']

    model_ids: iterable of strings, to be called by
    models.reference_cluster_slow_sequence.  Any of:

        ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X', 'Praesepe',
        'NGC-6811', '120-Myr', '300-Myr', '2.6-Gyr']

    poly_order: integer polynomial order for the reference cluster mean model.
    Default is 7, which seems to do the best job across all clusters.

    slow_seq_ages: optional list of ages, in Myr, for slow sequence models
    to underplot (e.g., [120, 150, 200, 250]).
    """
    # Get data
    N_colors = 6
    d = _get_cluster_Prot_Teff_data(N_colors=N_colors)

    # Make plot
    set_style("science")

    fig, ax = plt.subplots()

    for reference_cluster in reference_clusters:

        df = d[reference_cluster][0]
        color = d[reference_cluster][1]
        label = d[reference_cluster][2]
        zorder = d[reference_cluster][3]

        sel = df.flag_benchmark_period

        ax.scatter(
            df[sel].Teff_Curtis20, df[sel].Prot, color=color, alpha=1,
            s=25, rasterized=False, label=label, marker='o', edgecolors='k',
            linewidths=0.3, zorder=zorder
        )

        if show_binaries:

            sel = df.flag_possible_binary & ~pd.isnull(df.Teff_Curtis20)

            ax.scatter(
                df[sel].Teff_Curtis20, df[sel].Prot, color=color, alpha=0.5,
                s=12, rasterized=False, marker='o', edgecolors='k',
                linewidths=0., zorder=zorder-1
            )

    if isinstance(model_ids, list):
        Teff = np.linspace(3800, 6200, int(1e3))
        for model_id in model_ids:
            color = d[model_id][1]
            Prot = reference_cluster_slow_sequence(
                Teff, model_id, poly_order=poly_order
            )
            ax.plot(
                Teff, Prot, color=color, linewidth=1, zorder=-1
            )

    if isinstance(slow_seq_ages, list):
        Teff = np.linspace(3800, 6200, 100)
        for slow_seq_age in slow_seq_ages:
            Prot = slow_sequence(
                Teff, slow_seq_age, poly_order=poly_order
            )
            ax.plot(
                Teff, Prot, color='lightgray', linewidth=1, zorder=-1
            )

    ax.legend(loc='upper left', fontsize='x-small', handletextpad=0.1,
              borderaxespad=1., borderpad=0.5, fancybox=True, framealpha=0.8,
              frameon=False)

    ax.set_xlabel("Effective Temperature [K]")
    ax.set_ylabel("Rotation Period [days]")

    ax.set_xlim([7100, 2900])
    ax.set_xticks([7000, 6000, 5000, 4000, 3000])
    minor_xticks = np.arange(3000, 7100, 100)[::-1]
    ax.set_xticks(minor_xticks, minor=True)

    ax.set_ylim([-0.5, 16])
    ax.set_yticks([0, 5, 10, 15])
    if 'Ruprecht-147' in reference_clusters:
        ax.set_ylim([-0.5, 28])
        ax.set_yticks([0, 5, 10, 15, 20, 25])

    _given_ax_append_spectral_types(ax)

    basename = "_".join(reference_clusters)
    s = ''
    if show_binaries:
        s += '_showbinaries'
    b = ''
    if len(reference_clusters) == 1:
        b = 'singlecluster_'
    m = ''
    if isinstance(model_ids, list):
        m = f"_models_poly{poly_order}_" + "_".join(model_ids)
    ss = ''
    if isinstance(slow_seq_ages, list):
        slow_seq_ages = np.array(slow_seq_ages).astype(str)
        m = f"_slowseq_poly{poly_order}_" + "_".join(slow_seq_ages)

    outpath = join(outdir, f'{b}prot_vs_teff_{basename}{s}{m}{ss}.png')

    savefig(fig, outpath, dpi=400, writepdf=1)


def plot_prot_vs_teff_residual(
    outdir, reference_clusters, model_ids, poly_order=7
    ):
    """
    reference_clusters: list containing any of
        ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X', 'Praesepe',
        'NGC-6811', 'NGC-6819', 'Ruprecht-147']

    model_ids: iterable of strings, to be called by
    models.reference_cluster_slow_sequence.  Any of:

        ['Praesepe', 'NGC-6811', '120-Myr', '300-Myr', '2.6-Gyr']

    poly_order: integer polynomial order for the reference cluster mean model.
    Default is 7, which seems to do the best job across all clusters.
    """

    allowed_model_ids = [
        'Praesepe', 'NGC-6811', '120-Myr', '300-Myr', '2.6-Gyr'
    ]
    for model_id in model_ids:
        if model_id not in allowed_model_ids:
            errmsg = f"Got model_id {model_id} - not implemented in this plot!"
            raise ValueError(errmsg)

    # Get data
    d = _get_cluster_Prot_Teff_data()

    # Make plot
    set_style("clean")

    # Each mean model gets its own (data-model) vs Teff axis
    factor = 0.8
    fig = plt.figure(figsize=(factor*3.3*2, factor*1.5*2.5))
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )
    axs = [axd['A'], axd['B'], axd['C'], axd['D']]

    for ax, model_id in zip(axs, model_ids):

        _plot_prot_vs_teff_residual(
            ax, model_id, d, reference_clusters, poly_order
        )

    fig.text(-0.01, 0.5, "Rotation Period Data - Model [days]", va='center',
             rotation=90, fontsize='large')
    fig.text(0.5, -0.01, "Effective Temperature [K]", ha='center',
             fontsize='large')

    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    basename = "_".join(reference_clusters)
    b = ''
    if len(reference_clusters) == 1:
        b = 'singlecluster_'
    m = ''
    if isinstance(model_ids, list):
        m = f"_models_poly{poly_order}_" + "_".join(model_ids)

    outpath = join(outdir, f'{b}prot_vs_teff_residual_{basename}{m}.png')

    savefig(fig, outpath, dpi=400, writepdf=False)


def plot_cdf_fast_slow_ratio(
    outdir, poly_order=7,
    model_ids = ['α Per', '120-Myr', '300-Myr', 'Praesepe'],
    reference_clusters = ['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                          'Group-X', 'Praesepe', 'NGC-6811']
    ):
    """
    Plot: RATIO of cumulative distribution functions, showing the cdf(Teff) for
    the fast and slow sequence at various times.

    poly_order: integer polynomial order for the reference cluster mean model.
    Default is 7, which seems to do the best job across all clusters.
    """

    # model_ids: iterable of strings, to be called by
    # models.reference_cluster_slow_sequence.

    # Get data
    d = _get_cluster_Prot_Teff_data()

    # Make plot
    set_style("clean")

    # Each mean model gets its own (data-model) vs Teff axis
    factor = 0.8
    fig = plt.figure(figsize=(factor*3.3*3, factor*1.5*1.5*2.5))
    axd = fig.subplot_mosaic(
        """
        ABC
        012
        345
        """
    )
    axs = [axd['A'], axd['B'], axd['C']]

    ix = 0
    for ax, model_id in zip(axs, model_ids):

        # Get the data
        sel_teff_range = [3800, 6200]
        if model_id == 'α Per':
            set_toplot = {'α Per'}
        elif model_id == '120-Myr':
            set_120myr = {k for k,v in d.items() if "120" in v[2]}
            set_toplot = set_120myr.intersection(set(reference_clusters))
            sel_teff_range = [4500, 6200]
        elif model_id == '300-Myr':
            set_300myr = {k for k,v in d.items() if "300" in v[2]}
            set_toplot = set_300myr.intersection(set(reference_clusters))
        elif model_id == 'Praesepe':
            set_toplot = {'Praesepe'}
        elif model_id == 'NGC-6811':
            set_toplot = {'NGC-6811'}

        # collect effective temperatures for both slow and fast sequences, and
        # the model-ids specified above (default 120 Myr, 300 Myr,
        # 670Myr/Praesepe)
        teff_ss, teff_fs = [], []

        for reference_cluster in set_toplot:

            df = d[reference_cluster][0]
            color = d[reference_cluster][1]
            label = d[reference_cluster][2]
            zorder = d[reference_cluster][3]

            sel = df.flag_benchmark_period

            Teff = nparr(df[sel].Teff_Curtis20)
            Prot = nparr(df[sel].Prot)
            Prot_model = reference_cluster_slow_sequence(
                Teff, model_id, poly_order=poly_order
            )
            Prot_residual = Prot - Prot_model

            # Selection for "slow sequence"
            sel_prot = (Prot_residual > -2) & (Prot_residual < 2)
            #sel_teff = (Teff > sel_teff_range[0]) & (Teff < sel_teff_range[1])
            sel_ss = sel_prot# & sel_teff

            # Selection for "fast sequence"
            sel_prot_fs = (Prot_residual < -2)
            #sel_teff_fs = (Teff > 3800) & (Teff < 6200)
            sel_fs = sel_prot_fs# & sel_teff_fs

            teff_ss.append(Teff[sel_ss])
            teff_fs.append(Teff[sel_fs])

        teff_ss = np.hstack(teff_ss)
        teff_fs = np.hstack(teff_fs)
        teff_all = np.hstack([teff_ss, teff_fs])

        teff_continuous_bins = np.linspace(3800, 6200, int(1e4))
        # 3800, 4150, 4500, 4850, 5200, 5550, 5900, 6250
        teff_chunk_bins = np.arange(3800, 6200+350, 350)

        #
        # cumulative count distributions
        #
        c_vals_ss, _, patches_ss = ax.hist(
            teff_ss, teff_continuous_bins, density=False, histtype='step',
            cumulative=-1, label='Slow seq.', linewidth=0.5, fill=False,
            color='C0'
        )
        c_vals_fs, _, patches_ss = ax.hist(
            teff_fs, teff_continuous_bins, density=False, histtype='step',
            cumulative=-1, label='Fast seq.', linewidth=0.5, fill=False,
            color='C1'
        )
        c_vals_all, _, patches_all = ax.hist(
            teff_all, teff_continuous_bins, density=False, histtype='step',
            cumulative=-1, label='Slow + Fast', linewidth=0.5, fill=False,
            color='C2'
        )
        ax.update({
            'ylabel': 'Count',
            'title': model_id + f" N={int(max(c_vals_all))}",
            'xlim': [6300, 3700],
        })

        custom_lines = [Line2D([0], [0], color='C0', lw=0.5),
                        Line2D([0], [0], color='C1', lw=0.5),
                        Line2D([0], [0], color='C2', lw=0.5)]
        ax.legend(custom_lines, ['Slow', 'Fast', 'Slow+Fast'],
                  loc='upper left', fontsize='x-small', handletextpad=0.3,
                  borderaxespad=1.0, borderpad=0.4)

        #
        # now HISTOGRAM
        #
        ax = axd[str(ix)]

        h_vals_ss, _, _ = ax.hist(
            teff_ss, teff_chunk_bins, density=False, histtype='step',
            cumulative=False, label='Slow seq.', linewidth=2, fill=False,
            color='C0', zorder=1
        )
        h_vals_fs, _, _ = ax.hist(
            teff_fs, teff_chunk_bins, density=False, histtype='step',
            cumulative=False, label='Fast seq.', linewidth=0.8, fill=False,
            color='C1', zorder=2
        )
        #h_vals_all, _, _ = ax.hist(
        #    teff_all, teff_chunk_bins, density=False, histtype='step',
        #    cumulative=False, label='Slow + Fast', linewidth=0.5, fill=False,
        #    color='C2'
        #)
        ax.update({
            'ylabel': 'Stars per 350 K bin',
            'xlim': [6300, 3700],
        })

        #
        # now plot the RATIO of slow to fast
        #
        ax = axd[str(ix+3)]

        midway = teff_chunk_bins[0:-1] + np.diff(teff_chunk_bins)/2
        num = h_vals_fs
        denom = (h_vals_fs+h_vals_ss)
        ratio = num/denom

        sigma_num = np.sqrt(num) / num
        sigma_num[np.isnan(sigma_num)] = 0
        sigma_denom = np.sqrt(denom) / denom
        sigma_denom[np.isnan(sigma_denom)] = 0

        sigma_ratio = np.sqrt( sigma_num**2 + sigma_denom**2 )
        sigma_ratio_abs = sigma_ratio * ratio
        sigma_ratio_abs_hi = sigma_ratio_abs
        sigma_ratio_abs_lo = sigma_ratio_abs

        # define upper limits
        sel = sigma_ratio_abs_hi == 0.
        sigma_ratio_abs_hi[sel] = sigma_denom[sel]

        sigma_plot = np.vstack([sigma_ratio_abs_lo, sigma_ratio_abs_hi])

        ax.errorbar(
            midway, ratio, yerr=sigma_plot, c='k', ls='--', marker='o'
        )
        ax.update({
            'ylabel': 'Fast/(Slow+Fast)',
            'ylim': [-0.03, 1.03],
            'xlim': [6300, 3700],
        })

        outdf = pd.DataFrame({
            'Teff_midpoints': midway,
            'ratio': h_vals_fs/(h_vals_fs+h_vals_ss),
            'ratio_err_lo': sigma_ratio_abs_lo,
            'ratio_err_hi': sigma_ratio_abs_hi,
            'count_fast_seq': h_vals_fs,
            'count_slow_seq': h_vals_ss,
        })
        csvpath = os.path.join(outdir, f'{model_id}_cdf_fast_slow_ratio_data.csv')
        outdf.to_csv(csvpath, index=False)
        print(f"Wrote {csvpath}")

        ix += 1

    fig.text(0.5, -0.01, "Effective Temperature [K]", ha='center',
             fontsize='large')

    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    basename = "_".join(reference_clusters)
    b = ''
    if len(reference_clusters) == 1:
        b = 'singlecluster_'
    m = ''
    if isinstance(model_ids, list):
        m = f"_models_poly{poly_order}_" + "_".join(model_ids)

    outpath = join(outdir, f'{b}cdf_fast_slow_ratio_{basename}{m}.png')

    savefig(fig, outpath, dpi=400, writepdf=False)


def _plot_prot_vs_teff_residual(
    ax, model_id, d, reference_clusters, poly_order, showtxt=1, tefflim_ss=1
    ):
    # helper function that plots the DATA (residual) for the different open
    # clusters, and their concatenations

    # Get the data
    sel_teff_range = [3800, 6200]
    if model_id == 'α Per':
        set_toplot = {'α Per'}
    elif model_id == '120-Myr':
        set_120myr = {k for k,v in d.items() if "120" in v[2]}
        set_toplot = set_120myr.intersection(set(reference_clusters))
        sel_teff_range = [4500, 6200]
    elif model_id == '300-Myr':
        set_300myr = {k for k,v in d.items() if "300" in v[2]}
        set_toplot = set_300myr.intersection(set(reference_clusters))
    elif model_id == 'Praesepe':
        set_toplot = {'Praesepe'}
    elif model_id == 'NGC-6811':
        set_toplot = {'NGC-6811'}
    elif model_id == '2.6-Gyr':
        set_2pt6gyr = {k for k,v in d.items() if "2.5" in v[2] or "2.7" in v[2]}
        set_toplot = set_2pt6gyr.intersection(set(reference_clusters))

    for reference_cluster in set_toplot:

        df = d[reference_cluster][0]
        color = d[reference_cluster][1]
        label = d[reference_cluster][2]
        zorder = d[reference_cluster][3]

        sel = df.flag_benchmark_period

        Teff = nparr(df[sel].Teff_Curtis20)
        Prot = nparr(df[sel].Prot)
        Prot_model = reference_cluster_slow_sequence(
            Teff, model_id, poly_order=poly_order
        )
        Prot_residual = Prot - Prot_model

        # Evalute the width of the "good" part of the residual
        sel_prot = (Prot_residual > -2) & (Prot_residual < 2)
        sel_teff = (Teff > sel_teff_range[0]) & (Teff < sel_teff_range[1])
        if tefflim_ss:
            sel = sel_teff & sel_prot
        else:
            sel = sel_prot
        _prot_good_chunk = Prot_residual[sel]
        N_good_chunk = len(_prot_good_chunk)
        N_outlier = len(Prot_residual[sel_teff & ~sel_prot])
        N_tot_sel_teff = len(Prot_residual[sel_teff])
        N_tot_not_sel_teff = len(Prot_residual[~sel_teff])
        std_good_chunk = np.std(_prot_good_chunk)
        mad_good_chunk = np.median(
            np.abs(_prot_good_chunk - np.median(_prot_good_chunk))
        )
        msg = (
            f'poly {poly_order}.\n'
            f'cluster: {reference_cluster}.\n'
            f'Total stars in Teff {sel_teff_range[0]}-{sel_teff_range[1]}: {N_tot_sel_teff}.\n'
            f'Ngood={N_good_chunk}.  Noutlier={N_outlier}.\n'
            f'Nnotconverged={N_tot_not_sel_teff}.\n'
            f'std_Prot: {std_good_chunk:.3f} d,  mad_good_chunk: {mad_good_chunk:.3f} d.'
        )
        print(msg)

        age_txt = " ".join(label.split(" ")[:-1])
        if tefflim_ss:
            label_txt = (
                label.split(" ")[-1] +
                " (σ$_\mathrm{slow}$ = " + f"{std_good_chunk:.2f} d)"
            )
        else:
            label_txt = label.split(" ")[-1]
            if 'α Per' in label:
                label_txt = 'α Per'

        ax.scatter(
            Teff[sel], Prot_residual[sel], color=color, alpha=1, s=15,
            rasterized=False, label=label_txt, marker='o', edgecolors='k',
            linewidths=0.3, zorder=zorder
        )
        ax.scatter(
            Teff[~sel], Prot_residual[~sel], color=color, alpha=0.8, s=15,
            rasterized=False, marker='s', edgecolors='k', linewidths=0.3,
            zorder=zorder-1
        )

        if model_id in ['α Per', '120-Myr', '300-Myr']:
            _Teff = np.linspace(3800, 6200, 1000)
            _Prot_model = reference_cluster_slow_sequence(
                _Teff, model_id, poly_order=poly_order
            )
            ax.plot(
                _Teff, -_Prot_model,
                color='lightgray', linewidth=1, zorder=-1
            )

    ax.hlines(
        0, 1000, 10000, colors='darkgray', alpha=1,
        linestyles='-', zorder=-2, linewidths=0.6
    )

    ax.legend(loc='lower left', fontsize='x-small', handletextpad=0.1,
              borderaxespad=1.0, borderpad=0.4)

    if showtxt:
        ax.text(0.97, 0.97, age_txt, transform=ax.transAxes,
                ha='right',va='top', color='k')

    ax.set_xlim([6600, 3400])
    ax.set_xticks([6000, 5000, 4000])
    ax.set_xticklabels([6000, 5000, 4000])

    ax.set_ylim([-14, 6])
    ax.set_yticks([-10, -5, 0, 5])


def _get_model_histogram(age, bounds_error='limit', parameters='default'):

    ymin, ymax = -14, 6
    teffmin, teffmax = 3800, 6200
    y_grid = np.linspace(ymin, ymax, 1000)
    teff_grid = np.linspace(teffmin, teffmax, 1001)

    resid_y_Teff = slow_sequence_residual(
        age, y_grid=y_grid, teff_grid=teff_grid, bounds_error=bounds_error,
        parameters=parameters, verbose=False
    )

    teff_chunk_bins = np.arange(3800, 6200+350, 350)
    teff_midway = teff_chunk_bins[0:-1] + np.diff(teff_chunk_bins)/2

    hist_ss_vals = []
    hist_fs_vals = []
    for ix, teff_midpoint in enumerate(teff_midway):

        # Effective temperature subset
        teff_lo = teff_chunk_bins[ix]
        teff_hi = teff_chunk_bins[ix+1]
        sel_teff = (teff_grid > teff_lo) & (teff_grid < teff_hi)

        # "Slow sequence" selection
        sel_ss = (y_grid > -2) & (y_grid < 2)

        # "Fast sequence" selection
        sel_fs = (y_grid < -2)

        N_ss_teff = np.trapz(resid_y_Teff[sel_ss, :], y_grid[sel_ss], axis=0)
        N_ss = np.trapz(N_ss_teff[sel_teff], teff_grid[sel_teff])
        hist_ss_vals.append(N_ss)

        N_fs_teff = np.trapz(resid_y_Teff[sel_fs, :], y_grid[sel_fs], axis=0)
        N_fs = np.trapz(N_fs_teff[sel_teff], teff_grid[sel_teff])
        hist_fs_vals.append(N_fs)

    return np.array(hist_ss_vals), np.array(hist_fs_vals), teff_midway


def plot_data_vs_model_prot(
    outdir, poly_order=7, parameters='default',
    model_ids = ['120-Myr', '300-Myr', 'Praesepe'],
    reference_clusters = ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                          'Group-X', 'Praesepe', 'NGC-6811']
    ):
    """
    9 (or 12)-panel plot.  Requires: having previously run plot_cdf_fast_slow_ratio.

    Top row: Data-Mean actual stars for 120Myr,300Myr,670Myr
        i.e., plot_prot_vs_teff_residual
    Middle row: The smooth 2d model, same ages
        i.e., plot_slow_sequence_residual
    Bottom row: Comparison between data & smooth 2d model via dN/dTeff
        i.e. plot_cdf_fast_slow_ratio

    This is a dense plot that summarizes the entire modeling effort, as well as
    how good our model is doing!

    ----------
    model_ids: iterable of strings, to be called by
    models.reference_cluster_slow_sequence.
    """

    # Get data
    d = _get_cluster_Prot_Teff_data()

    # Make plot
    set_style("clean")

    # Each mean model gets its own (data-model) vs Teff axis
    factor = 0.8
    fig = plt.figure(figsize=(factor*3.0*3, factor*1.5*1.5*2.5))
    Ncols = len(model_ids)
    assert Ncols in [3,4]
    if Ncols == 3:
        axd = fig.subplot_mosaic(
            """
            012
            345
            678
            """
        )
    elif Ncols == 4:
        axd = fig.subplot_mosaic(
            """
            0123
            4567
            abcd
            """
        )

    fig.text(-0.01, 4/6-0.03, "Rotation Period Data - Model [days]", va='center',
             rotation=90, fontsize='large')

    fig.text(-0.01, 1/6+0.03/2, "Fast Fraction", va='center',
             rotation=90, fontsize='large')

    fig.text(0.5, -0.01, "Effective Temperature [K]", ha='center',
             fontsize='large')

    fig.tight_layout(h_pad=0.4, w_pad=0.4)


    #
    # TOP ROW (same as plot_prot_vs_teff_residual)
    #
    if Ncols == 3:
        axs = [axd['0'], axd['1'], axd['2']]
        if model_ids == ['α Per', '120-Myr', '300-Myr']:
            titles = ['80 Myr', '120 Myr', '300 Myr']
        elif model_ids == ['120-Myr', '300-Myr', 'Praesepe']:
            titles = ['120 Myr', '300 Myr', '670 Myr']
    elif Ncols == 4:
        axs = [axd['0'], axd['1'], axd['2'], axd['3']]
        titles = ['80 Myr', '120 Myr', '300 Myr', '670 Myr']

    for ax, model_id, title in zip(axs, model_ids, titles):
        _plot_prot_vs_teff_residual(
            ax, model_id, d, reference_clusters, poly_order, showtxt=0,
            tefflim_ss=0
        )
        ax.set_title(title)

    #
    # MIDDLE ROW (same as plot_slow_sequence_residual)
    #
    if Ncols == 3:
        axs = [axd['3'], axd['4'], axd['5']]
        if model_ids == ['α Per', '120-Myr', '300-Myr']:
            ages = [80, 120, 300]
        elif model_ids == ['120-Myr', '300-Myr', 'Praesepe']:
            ages = [120, 300, 670]
    elif Ncols == 4:
        axs = [axd['4'], axd['5'], axd['6'], axd['7']]
        ages = [80, 120, 300, 670]
    resid_Teffs = []
    bounds_error = 'limit'
    for ax, age in zip(axs, ages):
        resid_Teffs, teff_grid = _plot_slow_sequence_residual(
            fig, ax, age, bounds_error, resid_Teffs, showtxt=0, showcolorbar=0,
            parameters=parameters
        )

    #
    # FINAL ROW: data vs the model
    #
    if Ncols == 3:
        axs = [axd['6'], axd['7'], axd['8']]
    elif Ncols == 4:
        axs = [axd['a'], axd['b'], axd['c'], axd['d']]
    cachedir = os.path.join(RESULTSDIR, 'cdf_fast_slow_ratio')

    chi_sqs = []
    for ax, age, model_id in zip(axs, ages, model_ids):

        csvpath = os.path.join(RESULTSDIR, 'cdf_fast_slow_ratio',
                               f'{model_id}_cdf_fast_slow_ratio_data.csv')
        df = pd.read_csv(csvpath)

        _f = 1/0.5323 # fudge factor to yield red-chi^2 near unity
        logf_ml = -0.148
        if age in [80, 120, 300]:
            sigma = 0.1 * _f**(-0.5) * np.exp(logf_ml) # uniform weighting across the 7 bins
        elif age == 670:
            sigma = 0.01 * _f**(-0.5) * np.exp(logf_ml) # stricter requirement -- want it gonezo.

        #ax.plot(df.Teff_midpoints, df.ratio, c='k', ls='--', marker='*',
        #        label='Data', zorder=2)
        ax.errorbar(
            df.Teff_midpoints, df.ratio, yerr=sigma, c='k', ls='--',
            marker='o', ms=4.5, label='Data'
        )

        # NOTE: could be plotted continuously, rather than at these specific
        # teff bins, and ages.  for apples-to-apples, this is fine.
        h_vals_ss, h_vals_fs, teff_midway = _get_model_histogram(
            age, parameters=parameters
        )
        model_ratio = h_vals_fs / (h_vals_fs + h_vals_ss)
        ax.plot(teff_midway, model_ratio, c='gray', ls='-', marker='X',
                label='Best-fit Model', zorder=1, ms=4, lw=1, mew=0)

        pklpath = os.path.join(LOCALDIR, "gyroemp", "fitgyro_emcee_v00",
                               "fit_120-Myr_300-Myr_Praesepe.pkl")
        with open(pklpath, 'rb') as f:
            d = pickle.load(f)
            flat_samples = d['flat_samples']

        np.random.seed(42)
        N_to_show = 64
        sel_samples = flat_samples[
            np.random.choice(flat_samples.shape[0], N_to_show, replace=False)
        ]
        sigma_period = 0.51
        for ix in range(N_to_show):
            if ix % 8 == 0:
                print(age, ix)
            sample = sel_samples[ix, :]
            #C, C_y0, logk0, logk2, logf = theta
            parameters = {
                'A': 1,
                'B': 0,
                'C': sample[0],
                'C_y0': sample[1],
                'logk0': sample[2],
                'logk2': sample[3],
                'l1': -2*sigma_period,
                'k1': np.pi # a joke, but it works
            }
            h_vals_ss, h_vals_fs, teff_midway = _get_model_histogram(
                age, parameters=parameters
            )
            model_ratio = h_vals_fs / (h_vals_fs + h_vals_ss)
            ax.plot(teff_midway, model_ratio, c='darkgray', ls='-', marker='o',
                    zorder=-100, ms=2, alpha=0.1, lw=0.5, mew=0)

            #legend trick
            if ix == 0:
                ax.plot(teff_midway+9999, model_ratio, c='darkgray', ls='-', marker='o',
                        zorder=-100, ms=2, alpha=0.5, lw=0.5, mew=0,
                        label='Samples')

        ax.legend(loc='upper left', fontsize='x-small', handletextpad=0.5,
                  borderaxespad=1.0, borderpad=0.4)
        ax.set_ylim([-0.03, 1.03])

    # fix xlims
    for _, ax in axd.items():
        ax.set_xlim([6300, 3700])

    basename = "_".join(reference_clusters)
    b = ''
    if len(reference_clusters) == 1:
        b = 'singlecluster_'
    m = ''
    if isinstance(model_ids, list):
        m = f"_models_poly{poly_order}_" + "_".join(model_ids)

    outpath = join(outdir, f'data_vs_model_{basename}{m}.png')
    savefig(fig, outpath, dpi=400, writepdf=1)


def _plot_slow_sequence_residual(
    fig, ax, age, bounds_error, resid_Teffs, showtxt=1, showcolorbar=1,
    parameters='default'
    ):

    ymin, ymax = -14, 6
    teffmin, teffmax = 3800, 6200
    y_grid = np.linspace(ymin, ymax, 1000)
    teff_grid = np.linspace(teffmin, teffmax, 1001)

    resid_y_Teff = slow_sequence_residual(
        age, y_grid=y_grid, teff_grid=teff_grid, bounds_error=bounds_error,
        verbose=False, parameters=parameters
    )

    if age in [120, 300]:
        if age == 120:
            teff_limit = 4500
            # ratio from /doc/20220919_width_of_slow_sequence.txt should be
            # ~=13/127
            num, denom = 13, 127

        elif age == 300:
            teff_limit = 3800
            # ratio higher at 300 myr, since we're shifting the definition of 
            # where the slow sequence ends
            num, denom = 26, 133

        sel = (teff_grid > teff_limit)
        resid_ygrid = np.trapz(resid_y_Teff[:,sel], teff_grid[sel], axis=1)

        # sum of all counts within the "fast sequence" region
        sel = (y_grid < -2)
        s0 = np.trapz(resid_ygrid[sel], y_grid[sel], axis=0)

        # sum of all counts in the slow sequence
        s1 = np.trapz(resid_ygrid[~sel], y_grid[~sel], axis=0)

        r_obs = num/denom
        r_model = s0/(s0+s1)

        msg = (
            f'age {age} myr. teff > {teff_limit}. '
            f'r_obs = {num}/{denom} = {r_obs:.2f}. '
            f'r_model = {r_model:.2f}. '
        )
        print(msg)

    resid_Teff = np.trapz(resid_y_Teff, y_grid, axis=0)
    resid_Teffs.append(resid_Teff)

    norm = LogNorm(vmin=1e-2, vmax=1)
    #norm = Normalize(vmin=0, vmax=1)
    _p = ax.imshow(
        resid_y_Teff,
        extent=(teffmin, teffmax, ymin, ymax),
        aspect='auto',
        cmap=cm.Greys,
        origin='lower',
        norm=norm
    )

    if age == 670:
        x0,y0,dx,dy = 0.5, 0.2, 0.4, 0.1

        axins1 = inset_axes(ax, width="100%", height="100%",
                            # x0,y0, dx, dy
                            bbox_to_anchor=(x0,y0,dx,dy),
                            loc='lower left',
                            bbox_transform=ax.transAxes)

        ticks = [0.01, 0.1, 1]
        cb = fig.colorbar(_p, cax=axins1, orientation="horizontal",
                          ticks=ticks)
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize='small')
        cb.ax.set_xticklabels([0.01, 0.1, 1])
        cb.ax.set_title("Probability", fontsize='small', pad=0.1)


    ax.set_xlim([6600, 3400])
    ax.set_xticks([6000, 5000, 4000])
    ax.set_xticklabels([6000, 5000, 4000])

    ax.set_ylim([-14, 6])
    ax.set_yticks([-10, -5, 0, 5])

    if age < 1000:
        age_txt = str(age) + ' Myr'
    else:
        age_txt = str(int(age/1000)) + ' Gyr'

    if showtxt:
        ax.text(0.97, 0.97, age_txt, transform=ax.transAxes,
                ha='right', va='top', color='k')

    if showcolorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.08)
        fig.colorbar(_p, cax=cax, extend='both')

    return resid_Teffs, teff_grid


def plot_slow_sequence_residual(outdir, ages, bounds_error='nan'):

    # Each mean model gets its own (data-model) vs Teff axis
    set_style('clean')

    N_ages = len(ages)

    factor = 0.8
    xfactor = 1 if N_ages == 4 else 1.5

    fig = plt.figure(figsize=(xfactor*factor*3.3*2, factor*1.5*2.5))
    if N_ages == 4:
        axd = fig.subplot_mosaic(
            """
            AB
            CD
            """
        )
        axs = [axd['A'], axd['B'], axd['C'], axd['D']]
    elif N_ages == 6:
        axd = fig.subplot_mosaic(
            """
            ABC
            DEF
            """
        )
        axs = [axd['A'], axd['B'], axd['C'], axd['D'], axd['E'], axd['F']]

    resid_Teffs = []
    for ax, age in zip(axs, ages):

        resid_Teffs, teff_grid = _plot_slow_sequence_residual(
            fig, ax, age, bounds_error, resid_Teffs
        )

    fig.text(-0.01, 0.5, "Rotation Period Residual [days]", va='center',
             rotation=90, fontsize='large')
    fig.text(0.5, -0.01, "Effective Temperature [K]", ha='center',
             fontsize='large')

    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    b = "_".join(np.array(ages).astype(str))

    outpath = join(outdir, f'slow_sequence_residual_{b}.png')
    savefig(fig, outpath, writepdf=False)

    #
    # Part 2: integrated over the residual
    #
    plt.close("all")

    fig = plt.figure(figsize=(xfactor*factor*3.3*2, factor*1.5*2.5))
    if N_ages == 4:
        axd = fig.subplot_mosaic(
            """
            AB
            CD
            """
        )
        axs = [axd['A'], axd['B'], axd['C'], axd['D']]
    elif N_ages == 6:
        axd = fig.subplot_mosaic(
            """
            ABC
            DEF
            """
        )
        axs = [axd['A'], axd['B'], axd['C'], axd['D'], axd['E'], axd['F']]

    for ax, age, resid_Teff in zip(axs, ages, resid_Teffs):

        ax.plot(
            teff_grid, resid_Teff, lw=1, c='k'
        )

        ax.set_xlim([6600, 3400])
        ax.set_xticks([6000, 5500, 5000, 4500, 4000])

        if age < 1000:
            age_txt = str(age) + ' Myr'
        else:
            age_txt = str(int(age/1000)) + ' Gyr'

        ax.text(0.97, 0.97, age_txt, transform=ax.transAxes,
                ha='right',va='top', color='k')

    fig.text(-0.01, 0.5, "Relative probability", va='center',
             rotation=90, fontsize='large')
    fig.text(0.5, -0.01, "Effective Temperature [K]", ha='center',
             fontsize='large')

    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    b = "_".join(np.array(ages).astype(str))
    e = "_limitbounds" if bounds_error == 'limit' else ""

    outpath = join(outdir, f'ymarginalized_slow_sequence_residual_{b}{e}.png')
    savefig(fig, outpath, writepdf=False)



def plot_sub_praesepe_selection_cut(mdf5, poly_order=7):
    """
    make figure showing which stars have rotation periods that are being
    selected via "sub-praesepe"
    (called from drivers/build_koi_table.py)
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(4,3))

    csvpath = join(
        DATADIR, 'interim', 'slow_sequence_manual_selection',
        'Praesepe_slow_sequence.csv'
    )
    df_prae = pd.read_csv(csvpath)

    teff_model = np.arange(3800, 6200+1, 1)
    prot_model = reference_cluster_slow_sequence(
        teff_model, "Praesepe", poly_order=poly_order
    )

    ax.scatter(
        df_prae["Teff"], df_prae["Prot"], c='C0', marker='o', zorder=4,
        s=3, linewidths=0.5, label='Praesepe'
    )
    ax.plot(
        teff_model, prot_model, c='C0', ls='-', zorder=3, lw=1
    )

    period_cols = ['s19_Prot', 's21_Prot', 'm14_Prot', 'm15_Prot']

    for period_col in period_cols:
        # all KOIs
        if period_col == 's19_Prot':
            ax.scatter(
                mdf5["adopted_Teff"], mdf5[period_col], c='darkgray', marker='o',
                zorder=0, s=1, linewidths=0, label='KOIs'
            )
        else:
            # legend label trick
            ax.scatter(
                mdf5["adopted_Teff"], mdf5[period_col], c='darkgray', marker='o',
                zorder=0, s=1, linewidths=0
            )

        # KOIs below Praesepe
        sel_teff_range = (
            (mdf5["adopted_Teff"] >= 3800)
            &
            (mdf5["adopted_Teff"] <= 6200)
        )

        period_val = mdf5[period_col][sel_teff_range]
        teff_val = mdf5["adopted_Teff"][sel_teff_range]

        sel_below_Praesepe = (
            period_val < reference_cluster_slow_sequence(
                teff_val, "Praesepe", poly_order=poly_order
            )
        )

        if period_col == 's19_Prot':
            ax.scatter(
                teff_val[sel_below_Praesepe], period_val[sel_below_Praesepe],
                c='C1', marker='o', zorder=4, linewidths=0.5, s=2,
                label='KOIs below Praesepe'
            )
        else:
            # legend label trick
            ax.scatter(
                teff_val[sel_below_Praesepe], period_val[sel_below_Praesepe],
                c='C1', marker='o', zorder=4, linewidths=0.5, s=2,
            )

    ax.legend(loc='upper left', fontsize='xx-small', framealpha=1)

    ax.update({
        'xlabel': 'B+20 Teff',
        'ylabel': 'Prot [M14,M15,S19,S21]',
        'ylim': [0,20],
        'xlim': [6500, 3600]
    })

    outdir = join(RESULTSDIR, 'debug')
    if not os.path.exists(outdir): os.mkdir(outdir)
    outpath = join(
        outdir, f"koi_table_sub_praesepe_selection_cut_poly{poly_order}.png"
    )
    fig.savefig(outpath, dpi=350, bbox_inches='tight')


def plot_age_posteriors(
    Prots, Teff, outdir,
    age_grid=np.linspace(0, 2600, 500),
    bounds_error='limit'
    ):

    #
    # Calculate the age posterior
    #

    dfs, summary_dfs = [], []
    for Prot in Prots:
        Protstr = str(Prot).zfill(2)
        if bounds_error == 'nan':
            typestr = 'defaultgrid'
        elif bounds_error == 'limit':
            typestr = 'limitgrid'
        cachepath = os.path.join(outdir, f"Prot{Protstr}_Teff{Teff}_{typestr}.csv")
        if not os.path.exists(cachepath):
            age_post = gyro_age_posterior(
                Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
                verbose=False
            )
            df = pd.DataFrame({
                'age_grid': age_grid,
                'age_post': age_post
            })
            df.to_csv(cachepath)
            print(f"Wrote {cachepath}")

        df = pd.read_csv(cachepath)
        age_grid = np.array(df.age_grid)
        age_post = np.array(df.age_post)

        d = given_grid_post_get_summary_statistics(age_grid, age_post)
        d['Prot'] = Prot
        d['Teff'] = Teff
        summary_df = pd.DataFrame(d, index=[0])

        dfs.append(df)
        summary_dfs.append(summary_df)

    #
    # make plot
    #

    outpath = os.path.join(
        outdir,
        f"age_posteriors_Teff{str(Teff)}_Prot{'_'.join(np.array(Prots).astype(str))}.png"
    )

    #
    # Plot it
    #
    plt.close("all")
    set_style('clean')
    fig, ax = plt.subplots(figsize=(3.3, 2))

    N_colors = len(Prots)
    #cmap = cm.viridis(np.linspace(0,1,N_colors))
    #cmap = cm.Spectral(np.linspace(0,1,N_colors))
    cmap = cm.cividis(np.linspace(0,1,N_colors))

    for ix, Prot in enumerate(Prots):

        df = dfs[ix]
        color = cmap[ix]

        #label = r"P$_{\rm rot}=$"+f"{Prot:.1f}d"
        label = f"{Prot:.1f}"+"$\,$d"

        ax.plot(df.age_grid, 1e3*df.age_post, color=color, ls='-', lw=1,
                label=label)

    ax.legend(loc='best', fontsize='small', handletextpad=0.2,
              borderaxespad=1., borderpad=0.5, fancybox=True, framealpha=0.8,
              frameon=False)

    ax.update({
        'xlabel': 'Age [Myr]',
        'ylabel': 'Probability (Myr$^{-1}$)',
        'xlim': [0, 1700],
    })
    ax.set_title(f'{Teff}'+'$\,$K', pad=-4)

    savefig(fig, outpath, dpi=400, writepdf=1)



def plot_age_posterior(
    Prot, Teff, outdir,
    age_grid=np.linspace(0, 2600, 500),
    bounds_error='limit'
    ):

    #
    # Calculate the age posterior
    #
    Protstr = str(Prot).zfill(2)
    if bounds_error == 'nan':
        typestr = 'defaultgrid'
    elif bounds_error == 'limit':
        typestr = 'limitgrid'
    cachepath = os.path.join(outdir, f"Prot{Protstr}_Teff{Teff}_{typestr}.csv")
    if not os.path.exists(cachepath):
        age_post = gyro_age_posterior(
            Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
            verbose=False
        )
        df = pd.DataFrame({
            'age_grid': age_grid,
            'age_post': age_post
        })
        df.to_csv(cachepath)
        print(f"Wrote {cachepath}")

    df = pd.read_csv(cachepath)
    age_grid = np.array(df.age_grid)
    age_post = np.array(df.age_post)
    age_peak = int(age_grid[np.argmax(age_post)])

    if type(age_post[0]) != str:

        cumulative = np.cumsum(age_post)

        # normalize to 1
        cumulative /= cumulative[-1]

        # consider different possible estimators for age
        if np.sum(cumulative == 1) > 1:
            # well-covered PDF
            end_ix = np.argwhere(cumulative == 1)[0]
            sel = slice(0,int(end_ix)+1)
            fn = interp1d(cumulative[sel], age_grid[sel], kind='quadratic',
                          bounds_error=True, fill_value=np.nan)
        else:
            print("WRN! Got a pdf with an age grid that doesn't sample "
                  "all likely ages.")
            fn = interp1d(cumulative, age_grid, kind='quadratic',
                          bounds_error=True, fill_value=np.nan)

        if age_peak >= 120:
            age_med = int(fn(0.5))
            age_lower = int(age_med - fn(0.5-0.34))
            age_upper = int(fn(0.5+0.34) - age_med)
            agestr = f"age = {age_med} +{age_upper} -{age_lower} myr"
        elif age_peak < 120:
            age_twosig = int(fn(0.9545))
            age_threesig = int(fn(0.9973))
            agestr = f"age < {age_twosig} myr (2σ), age < {age_threesig} myr (3σ)"
        elif age_peak > 1000:
            age_twosig = int(fn(1-0.9545))
            age_threesig = int(fn(1-0.9973))
            agestr = f"age > {age_twosig} myr (2σ), age > {age_threesig} myr (3σ)"

        titlestr = (
            f'Teff = {Teff} +- 50.  Prot = {Prot} +- 0.05.\n'
            f'{agestr}'
        )
    else:
        titlestr = (
            f'Teff = {Teff} +- 50.  Prot = {Prot} +- 0.05.\n'
            f'age = {age_peak}.'
        )

    outpath = os.path.join(outdir, f"age_posterior_Teff{Teff}_Prot{Protstr}.png")

    #
    # Plot it
    #
    plt.close("all")
    set_style('clean')
    fig, ax = plt.subplots()

    ax.plot(age_grid, age_post, c='k', ls='-', lw=1)

    if isinstance(titlestr, str):
        ax.set_title(titlestr, fontsize='small')

    ax.update({
        'xlabel': 'Age [Myr]',
        'ylabel': 'Probability (Myr$^{-1}$)',
        'xlim': [0, 1500]
    })

    fig.savefig(outpath, dpi=350, bbox_inches='tight')


def _given_params_plot_imshow(ax, xkey, ykey, df, vmin, i, j, map_row):

    A_grid = np.array([1])
    B_grid = np.array([0])
    #C_grid = np.arange(1.1, 20.1, 0.1)
    #C_y0_grid = np.arange(0.2, 0.9, 0.01)
    #logk0_grid = np.arange(-8,2,0.1)
    #logk2_grid = np.arange(-8,-4.5,0.1)
    C_grid = np.arange(1.1, 20.1, 0.2)
    C_y0_grid = np.arange(0.4, 0.8, 0.02)
    logk0_grid = np.arange(-8, 3, 0.5)
    logk2_grid = np.arange(-8,-4.5,0.2)


    #A_grid = np.arange(0.1, 1.6, 0.1)
    #B_grid = np.array([0])
    #C_grid = np.arange(0.5, 4.1, 0.1)
    #C_y0_grid = np.arange(0.2, 0.9, 0.1)
    #logk0_grid = np.arange(-6,-3.9,0.2)
    #logk2_grid = np.arange(-8,-4.5,0.2)

    #A_grid = np.arange(0.1, 2.1, 0.2)
    #logB_grid = np.arange(-6, 2.0, 1.0)
    #B_grid = np.array([0])
    #C_grid = np.arange(0.5, 2.1, 0.2)
    #C_y0_grid = np.arange(0.2, 0.9, 0.1)
    #logk0_grid = np.arange(-6,-3.9,0.5)
    #logk2_grid = np.arange(-8,-3.9,0.5)

    grid_dict = {
        "A":A_grid,
        #"logB":logB_grid,
        "B":B_grid,
        "C":C_grid,
        "C_y0":C_y0_grid,
        "logk0":logk0_grid,
        "logk2":logk2_grid
    }

    min_chisq = np.zeros((len(grid_dict[xkey]), len(grid_dict[ykey])))

    for ix, x in enumerate(grid_dict[xkey]):
        for iy, y in enumerate(grid_dict[ykey]):

            sel = (
                (df[xkey].round(2) == np.round(x,2)) &
                (df[ykey].round(2) == np.round(y,2))
            )

            try:
                min_chisq[ix, iy] = np.nanmin(df.loc[sel, 'chi_sq_red'])
            except ValueError:
                pass

    if xkey != ykey:

        norm = Normalize(vmin=vmin, vmax=vmin*2)
        xmin, xmax = grid_dict[xkey].min(), grid_dict[xkey].max()
        ymin, ymax = grid_dict[ykey].min(), grid_dict[ykey].max()
        _p = ax.imshow(
            min_chisq.T,
            extent=(xmin, xmax, ymin, ymax),
            aspect='auto',
            cmap=cm.Greys,
            origin='lower',
            norm=norm
        )

        ax.scatter(
            map_row[xkey], map_row[ykey], s=10, marker='*', c='k', zorder=10
        )
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))

        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)

        showcolorbar = 0
        if showcolorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(_p, cax=cax)

    else:
        pass


def plot_fit_gyro_model(outdir, modelid):
    """
    Corner plot showing chi^2 surface
    """

    Bstr = 'logB' if 'zeroB' not in modelid else 'B'

    param_list = f"C,C_y0,logk0,logk2".split(',')
    N_params = len(param_list)

    csvpath = os.path.join(RESULTSDIR, "fit_gyro_model",
                           f'{modelid}_concatenated_chi_squared_results.csv')

    # from write-temp.py
    df = pd.read_csv(
        csvpath, names=f"A,{Bstr},C,C_y0,logk0,k1,l1,logk2,chi_sq_red,BIC,n,k".split(',')
    )
    df['logk1'] = np.log(df.k1)
    df = df.drop(['k1'], axis='columns')

    print(df.sort_values(by='chi_sq_red').head(n=20))

    map_row = df.sort_values(by='chi_sq_red').head(n=1)
    outpath = os.path.join(outdir, f"map_row_{modelid}.csv")
    map_row.to_csv(outpath, index=False)
    print(f"Wrote {outpath}")

    vmin = np.nanmin(df.chi_sq_red)

    # Make plot
    set_style("clean")

    # Each mean model gets its own (data-model) vs Teff axis
    fig, axs = plt.subplots(figsize=(8,8), nrows=N_params-1, ncols=N_params-1)

    for i in range(N_params):
        for j in range(N_params):
            xkey = param_list[i]
            ykey = param_list[j]
            if j>i:
                _given_params_plot_imshow(
                    axs[j-1,i], xkey, ykey, df, vmin, i, j, map_row
                )

    # clean up axes
    for i in range(N_params-1):
        for j in range(N_params-1):
            if i < j:
                axs[i,j].axis("off")

    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    outpath = join(outdir, f'fit_gyro_model_{modelid}.png')
    savefig(fig, outpath, dpi=400, writepdf=False)


def DEPRECATED_plot_fit_gyro_model(outdir, modelid):
    """
    DEPRECATED

    Corner plot showing chi^2 surface, sampling over A,B,C,C_y0,logk0,logk2.
    """

    Bstr = 'logB' if 'zeroB' not in modelid else 'B'

    param_list = f"A,{Bstr},C,C_y0,logk0,logk2".split(',')
    N_params = len(param_list)

    csvpath = os.path.join(RESULTSDIR, "fit_gyro_model",
                           f'{modelid}_concatenated_chi_squared_results.csv')

    # from write-temp.py
    df = pd.read_csv(
        csvpath, names=f"A,{Bstr},C,C_y0,logk0,k1,l1,logk2,chi_sq_red,BIC".split(',')
    )
    df['logk1'] = np.log(df.k1)
    df = df.drop(['k1'], axis='columns')

    print(df.sort_values(by='chi_sq_red').head(n=20))

    map_row = df.sort_values(by='chi_sq_red').head(n=1)
    outpath = os.path.join(outdir, f"map_row_{modelid}.csv")
    map_row.to_csv(outpath, index=False)
    print(f"Wrote {outpath}")

    vmin = np.nanmin(df.chi_sq_red)

    # TODO sanity check...
    DEBUG = 0
    if DEBUG:
        fig, ax = plt.subplots()
        _given_params_plot_imshow(
            ax, "A", "B", df, vmin, 0, 1, map_row
        )
        outpath = join(outdir, f'temp.png')
        savefig(fig, outpath, dpi=400, writepdf=False)

    # Make plot
    set_style("clean")

    # Each mean model gets its own (data-model) vs Teff axis
    fig, axs = plt.subplots(figsize=(8,8), nrows=N_params-1, ncols=N_params-1)

    for i in range(N_params):
        for j in range(N_params):
            xkey = param_list[i]
            ykey = param_list[j]
            if j>i:
                _given_params_plot_imshow(
                    axs[j-1,i], xkey, ykey, df, vmin, i, j, map_row
                )

    # clean up axes
    for i in range(N_params-1):
        for j in range(N_params-1):
            if i < j:
                axs[i,j].axis("off")

    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    outpath = join(outdir, f'fit_gyro_model_{modelid}.png')
    savefig(fig, outpath, dpi=400, writepdf=False)


def _get_empgyro_grid_data(imagestr, n, poly_order, age_scale):

    # from run_prot_teff_grid.py
    teffmin, teffmax = 3800, 6200
    protmin, protmax = 0, 23
    Teff_grid = np.arange(teffmin, teffmax+50, 50)
    Prot_grid = np.arange(protmin, protmax+0.5, 0.5)
    N_Teff = len(Teff_grid)
    N_Prot = len(Prot_grid)

    typestr = 'limitgrid'
    cachedir = os.path.join(
        LOCALDIR, "gyroemp",
        f"prot_teff_grid_n{n:.1f}_reluncpt1pct_{age_scale}"
    )
    _fpaths = [
        os.path.join(
            cachedir,
            f"Prot{float(Prot):.2f}_Teff{float(Teff):.1f}_{typestr}.csv"
        )
        for Prot, Teff in product(Prot_grid, Teff_grid)
    ]
    fpaths = [f for f in _fpaths if os.path.exists(f)]

    df = pd.concat(( pd.read_csv(f) for f in fpaths ))

    p1sig = np.zeros((N_Teff, N_Prot))
    m1sig = np.zeros((N_Teff, N_Prot))
    median = np.zeros((N_Teff, N_Prot))
    peak = np.zeros((N_Teff, N_Prot))

    for ix, x in enumerate(Teff_grid):
        for iy, y in enumerate(Prot_grid):

            sel = (
                (df['Teff'].round(2) == np.round(x,2)) &
                (df['Prot'].round(3) == np.round(y,3))
            )

            if 'abs' not in imagestr:
                p1sig[ix, iy] = df.loc[sel, '+1sigmapct']
                m1sig[ix, iy] = df.loc[sel, '-1sigmapct']
            else:
                p1sig[ix, iy] = df.loc[sel, '+1sigma']
                m1sig[ix, iy] = df.loc[sel, '-1sigma']
            median[ix, iy] = df.loc[sel, 'median']
            peak[ix, iy] = df.loc[sel, 'peak']

            if y > slow_sequence(
                x, 2600, poly_order=poly_order, n=n
            ):
                p1sig[ix, iy] = np.nan
                m1sig[ix, iy] = np.nan
                median[ix, iy] = np.nan
                peak[ix, iy] = np.nan

    return p1sig, m1sig, median, peak


def plot_empirical_limits_of_gyrochronology(
    outdir, imagestr, poly_order=7, n=0.5, age_scale='default',
    slow_seq_ages=None, writepdf=0
    ):
    """
    Map out precision of gyro posteriors as a function of Prot and Teff.

    args:

        poly_order: integer polynomial order for the reference cluster mean
        model.  Default is 7, which seems to do the best job across all
        clusters.

        slow_seq_ages: optional list of ages, in Myr, for slow sequence models
        to underplot (e.g., [100, 200, 400]).

        n: assume Prot ~ t^{n} scaling

        age_scale: "default", "1sigmaolder", or "1sigmayounger".  Shifts the
        entire age scale appropriately.
    """

    allowedstrs = [
        'plus', 'minus', 'plus_abs', 'minus_abs', 'median', 'peak',
        'both', 'both_abs', 'diff_median', 'diff_median_abs', 'diff_peak'
    ]
    assert imagestr in allowedstrs
    singleaxstrs = [s for s in allowedstrs if 'both' not in s]

    assert age_scale in ["default", "1sigmaolder", "1sigmayounger"]

    #
    # Get data
    #
    p1sig, m1sig, median, peak = _get_empgyro_grid_data(
        imagestr, n, poly_order, age_scale
    )
    if imagestr in ['diff_median', 'diff_median_abs', 'diff_peak']:
        npt5_p1sig, npt5_m1sig, npt5_median, npt5_peak = (
            _get_empgyro_grid_data(
                imagestr, 0.5, poly_order, 'default'
            )
        )
        dmedian = median - npt5_median
        dmedian_rel = (median - npt5_median) / npt5_median
        dpeak = peak - npt5_peak
        sel = np.abs(dmedian) > 25
        sel1 = np.abs(dmedian) > 40
        msg = (
            f"n={n}:\n"
            f"mean: {np.nanmean(dmedian):.3f}\n"
            f"mean of abs>25: {np.nanmean(dmedian[sel]):.3f}\n"
            f"mean of abs>40: {np.nanmean(dmedian[sel1]):.3f}\n"
            f"abs median: {np.nanmedian(np.abs(dmedian)):.3f}\n"
            f"abs 25th: {np.nanpercentile(np.abs(dmedian),25):.3f}\n"
            f"abs 75th: {np.nanpercentile(np.abs(dmedian),75):.3f}\n"
            f"abs 95th: {np.nanpercentile(np.abs(dmedian),95):.3f}\n"
            f"abs 99th: {np.nanpercentile(np.abs(dmedian),99):.3f}\n"
        )
        print(42*'-')
        print(msg)

    # Make plot
    set_style("science")

    # max width is 3.5 for single column
    if 'both' in imagestr:
        from matplotlib import gridspec
        figsize = (6.9, 4)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1,3, width_ratios=[4, 4, 0.2])
        axs = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])]
    else:
        fig, ax = plt.subplots()

    if imagestr in ['plus', 'minus', 'both']:
        #norm = Normalize(vmin=0., vmax=1)
        norm = LogNorm(vmin=10**-1.5, vmax=10**0.25) #0.0316 to 3.16
    elif imagestr in ['plus_abs', 'minus_abs', 'both_abs']:
        norm = LogNorm(vmin=50, vmax=500)
    elif imagestr in ['peak', 'median']:
        norm = LogNorm(vmin=10, vmax=2600)
    elif imagestr in ['diff_median_abs', 'diff_peak_abs']:
        norm = Normalize(vmin=-100, vmax=100)
    elif imagestr in ['diff_median', 'diff_peak']:
        if age_scale == 'default':
            norm = Normalize(vmin=-0.1, vmax=0.1)
        else:
            norm = Normalize(vmin=-0.2, vmax=0.2)

    if imagestr in ['plus', 'plus_abs']:
        img = p1sig.T
    elif imagestr in ['minus', 'minus_abs']:
        img = m1sig.T
    elif imagestr in ['both_abs', 'both']:
        img0 = p1sig.T
        img1 = m1sig.T
    elif imagestr in ['median']:
        img = median.T
    elif imagestr in ['peak']:
        img = peak.T
    elif imagestr in ['diff_median']:
        img = dmedian_rel.T
    elif imagestr in ['diff_median_abs']:
        img = dmedian.T
    elif imagestr in ['diff_peak']:
        img = dpeak.T

    if 'diff' not in imagestr:
        # sequential
        cmap = mpl.colormaps['plasma']
        cmap = mpl.cm.get_cmap("plasma", 7)
    else:
        # divering
        cmap = mpl.colormaps['bwr']
        if age_scale == 'default':
            cmap = mpl.cm.get_cmap("bwr", 8)
        else:
            cmap = mpl.cm.get_cmap("bwr", 16)
    #_cmap = cmap(np.arange(0,cmap.N))

    # # WHITE TOP OUTLIER
    # white = np.array([256/256, 256/256, 256/256, 1])
    # green = np.array([0/256, 256/256, 0/256, 1])
    # _cmap[-1, :] = white

    #_cmap[0, :] = green
    #newcmp = ListedColormap(_cmap)

    teffmin, teffmax = 3800, 6200
    protmin, protmax = 0, 23

    if imagestr in singleaxstrs:
        _p = ax.imshow(
            img,
            extent=(teffmin, teffmax, protmin, protmax),
            aspect='auto',
            cmap=cmap,
            origin='lower',
            norm=norm
        )
    elif 'both' in imagestr:
        _ = axs[0].imshow(
            img0,
            extent=(teffmin, teffmax, protmin, protmax),
            aspect='auto',
            cmap=cmap,
            origin='lower',
            norm=norm
        )
        l0 = '/$t$' if imagestr == 'both' else ''
        axs[0].text(0.03, 0.97, '+1$\sigma_t$'+l0, transform=axs[0].transAxes,
                    ha='left', va='top', color='k')

        _p = axs[1].imshow(
            img1,
            extent=(teffmin, teffmax, protmin, protmax),
            aspect='auto',
            cmap=cmap,
            origin='lower',
            norm=norm
        )
        axs[1].text(0.03, 0.97, '-1$\sigma_t$'+l0, transform=axs[1].transAxes,
                    ha='left', va='top', color='k')

    if 'both' not in imagestr:
        cb = fig.colorbar(_p, extend='neither')
    else:
        # left/bottom/width/height, fractions of figwidth/height
        cb = fig.colorbar(_p, cax=axs[-1], extend='neither')

    if 'plus' in imagestr:
        labelstr = "+"
    elif 'minus' in imagestr:
        labelstr = "-"
    elif 'both' in imagestr:
        labelstr = '$\pm$'

    if imagestr in ['plus', 'minus', 'both']:
        cb.set_label(labelstr + '$1\sigma_t/\mathrm{median}(t)$')
        if imagestr == 'both':
            cb.set_ticks([10**-1.5, 0.1, 10**-0.5, 1])
            cb.set_ticklabels([0.03, 0.1, 0.3, 1])
            cb.ax.minorticks_off()

    elif "abs" in imagestr and "diff" not in imagestr:
        cb.set_label(labelstr + '1$\sigma_t$ [Myr]')
        cb.set_ticks([50, 100, 200, 400, 500])
        cb.set_ticklabels([50, 100, 200, 400, 500])
        cb.ax.minorticks_off()
    elif imagestr in ['median', 'peak']:
        cb.set_label('$t$ [Myr]')
    elif imagestr in ['diff_median']:
        cb.set_label('$\Delta t / t$')
    elif imagestr in ['diff_median_abs', 'diff_peak']:
        cb.set_label('$\Delta t$ [Myr]')

    if imagestr in singleaxstrs:
        ax.set_ylim([0, 23])
        ax.set_yticks([0, 5, 10, 15, 20])
    elif 'both' in imagestr:
        for ax in axs[:-1]:
            ax.set_ylim([0, 23])
            ax.set_yticks([0, 5, 10, 15, 20])
        axs[1].set_yticklabels([])

    if isinstance(slow_seq_ages, (list, np.ndarray)) and (
        imagestr in singleaxstrs
    ):

        reference_ages = agedict['default']['reference_ages']

        Teff = np.linspace(3800, 6200, 100)
        for slow_seq_age in slow_seq_ages:
            Prot = slow_sequence(
                Teff, slow_seq_age, poly_order=poly_order, n=n,
                reference_ages=reference_ages
            )
            if slow_seq_age % 500 == 0:
                linewidth = 1.5
                linestyle = '-'
            else:
                linewidth = 0.5
                linestyle = ':'
            ax.plot(
                Teff, Prot, color='lightgray', linewidth=linewidth,
                linestyle=linestyle, zorder=999
            )
    if isinstance(slow_seq_ages, (list, np.ndarray)) and 'both' in imagestr:
        Teff = np.linspace(3800, 6200, 100)
        for slow_seq_age in slow_seq_ages:
            Prot = slow_sequence(
                Teff, slow_seq_age, poly_order=poly_order, n=n,
                reference_ages=reference_ages
            )
            if slow_seq_age % 500 == 0:
                linewidth = 1.5
                linestyle = '-'
            else:
                linewidth = 0.5
                linestyle = ':'
            for ax in axs[:-1]:
                ax.plot(
                    Teff, Prot, color='lightgray', linewidth=linewidth,
                    linestyle=linestyle, zorder=999
                )

    if imagestr in singleaxstrs:
        ax.set_xlabel("Effective Temperature [K]")
        ax.set_ylabel("Rotation Period [days]")

        ax.set_xlim([6200, 3800])
        ax.set_xticks([6000, 5000, 4000])
        minor_xticks = np.arange(3800, 6300, 100)[::-1]
        ax.set_xticks(minor_xticks, minor=True)

        _sptypes=['G2V','K0V','K5V','M0V']
        _given_ax_append_spectral_types(ax, _sptypes=_sptypes)
    elif 'both' in imagestr:
        axs[0].set_ylabel("Rotation Period [days]")
        for ax in axs[:-1]:
            ax.set_xlabel("Effective Temperature [K]")

            ax.set_xlim([6200, 3800])
            ax.set_xticks([6000, 5000, 4000])
            minor_xticks = np.arange(3800, 6300, 100)[::-1]
            ax.set_xticks(minor_xticks, minor=True)

            _sptypes=['G2V','K0V','K5V','M0V']
            _given_ax_append_spectral_types(ax, _sptypes=_sptypes)

    basename = "empirical_limits_of_gyrochronology"
    s = f'_n{n:.1f}'
    ss = f'_scale-{age_scale}'
    if isinstance(slow_seq_ages, (list, np.ndarray)):
        slow_seq_ages = np.array(slow_seq_ages).astype(str)
        m = f"_slowseq_poly{poly_order}_" + "_".join(slow_seq_ages)

    outpath = join(outdir, f'{imagestr}_{basename}{s}{m}{ss}.png')

    if 'both' in imagestr:
        fig.tight_layout(h_pad=0.4, w_pad=0.4)

    savefig(fig, outpath, dpi=400, writepdf=writepdf)
