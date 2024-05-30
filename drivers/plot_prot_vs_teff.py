import os
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR
import numpy as np

PLOTDIR = os.path.join(RESULTSDIR, 'prot_vs_teff')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

# figure 1a variants for talks
# ...with M67
all_clusters2 = ['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                 'Group-X', 'Praesepe', 'NGC-6811', 'NGC-6819', 'Ruprecht-147',
                 'M67']
gp.plot_prot_vs_teff(
    outdir, all_clusters2, slow_seq_ages=[80, 120, 300, 670, 1000, 2600, 4000]
)
slow_seq_ages = list(np.arange(250, 4250, 250))
gp.plot_prot_vs_teff(
    outdir, all_clusters2, slow_seq_ages=slow_seq_ages
)

gp.plot_prot_vs_teff(
    outdir, ['Blanco-1', 'Psc-Eri', 'Praesepe', 'NGC-6811'], smallfigsizex=1
)

# the standard <=2.7 gyr clusters
all_clusters1 = ['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                 'Group-X', 'Praesepe', 'NGC-6811', 'NGC-6819', 'Ruprecht-147']
slow_seq_ages = list(np.arange(100, 2700, 100))

# figure 1a variants for talks
gp.plot_prot_vs_teff(
    outdir, all_clusters1, slow_seq_ages=[80, 120, 300, 670, 1000, 2600]
)
gp.plot_prot_vs_teff(
    outdir, all_clusters1, slow_seq_ages=slow_seq_ages
)

# figure 1a
for logy in [0, 1]:
    gp.plot_prot_vs_teff(
        outdir, all_clusters1, slow_seq_ages=slow_seq_ages, logy=logy
    )

# figure 1b
all_clusters = ['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
                 'Praesepe', 'NGC-6811']
gp.plot_prot_vs_teff(
    outdir, all_clusters
)
model_ids = ['α Per', '120-Myr', '300-Myr', 'Praesepe', 'NGC-6811']
for logy in [0, 1]:
    gp.plot_prot_vs_teff(
        outdir, all_clusters, model_ids=model_ids, logy=logy
    )

assert 0

# just models
all_clusters1 = [None]
slow_seq_ages = [120+0.5*(300-120), 300+0.5*(670-300), 670+0.5*(1000-670)]
model_ids = ['120-Myr', '300-Myr', 'Praesepe', 'NGC-6811']
gp.plot_prot_vs_teff(
    outdir, all_clusters1, slow_seq_ages=slow_seq_ages,
    model_ids=model_ids, n=None, interp_method="skumanich_vary_n"
)

for interp_method in ["alt", "diff"]:
    for n in [0, 0.2, 0.5, 1.0, 2, 0.1]:
        gp.plot_prot_vs_teff(
            outdir, all_clusters1, slow_seq_ages=slow_seq_ages,
            model_ids=model_ids, n=n, interp_method=interp_method
        )
assert 0

# logo
all_clusters1 = ['Pleiades', 'Blanco-1', 'Psc-Eri', 'Praesepe', 'NGC-6811']
slow_seq_ages = [450]
model_ids = ['120-Myr', 'Praesepe', 'NGC-6811']
gp.plot_prot_vs_teff(
    outdir, all_clusters1, slow_seq_ages=slow_seq_ages, model_ids=model_ids,
    hide_ax=1, logo_colors=1
)
assert 0

# figure 3 base sketch
slow_seq_ages = [120, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
gp.plot_prot_vs_teff(
    outdir, [], slow_seq_ages=slow_seq_ages
)

##########################################

#
# single clusters, with reference fits
#
list0 = ['Pleiades', 'Psc-Eri', 'Blanco-1', 'NGC-3532', 'Praesepe', 'NGC-6811']
list1 = ['120-Myr', '120-Myr', '120-Myr', 'NGC-3532', 'Praesepe', 'NGC-6811']
for reference_cluster, model_id in zip(list0, list1):
    for poly_order in [5,6,7]:
        gp.plot_prot_vs_teff(outdir, [reference_cluster], show_binaries=1,
                             model_ids=[model_id], poly_order=poly_order)

#
# single clusters, with binarity shown
#
reference_cluster_sets = [
    ['Psc-Eri'],
    ['Pleiades'],
    ['Blanco-1'],
    ['NGC-3532'],
    ['Group-X'],
    ['Praesepe'],
    ['NGC-6811'],
]
for reference_clusters in reference_cluster_sets:
    gp.plot_prot_vs_teff(outdir, reference_clusters, show_binaries=1)

# Everything else
do_everything_else = 0
if do_everything_else:
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Praesepe', 'NGC-6811'])
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'Praesepe'])
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'Praesepe', 'NGC-6811'])
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'NGC-3532', 'Praesepe'])
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'Group-X', 'Praesepe'])
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'NGC-3532', 'Praesepe', 'NGC-6811'])
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'NGC-3532', 'Group-X', 'Praesepe', 'NGC-6811'])
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'Blanco-1', 'NGC-3532', 'Group-X', 'Praesepe', 'NGC-6811'])
    gp.plot_prot_vs_teff(outdir, ['Pleiades', 'NGC-3532', 'Group-X', 'NGC-6811'])


# chi2, bic, etc
clusters = [
    ['Pleiades', 'Blanco-1', 'Psc-Eri'],
    ['NGC-3532', 'Group-X'],
    ['Praesepe'],
    ['NGC-6811'],
    ['NGC-6819', 'Ruprecht-147']
]
model_ids = [
    ['120-Myr'],
    ['300-Myr'],
    ['Praesepe'],
    ['NGC-6811'],
    ['2.6-Gyr']
]
for poly_order in [2,3,4,5,6,7,8,9]:
    for _c, model_id in zip(clusters, model_ids):
        gp.plot_prot_vs_teff(outdir, _c,
                             model_ids=model_id,
                             poly_order=poly_order, show_resid=1)

