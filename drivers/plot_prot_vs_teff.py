import os
import gyrointerp.plotting as ap
from gyrointerp.paths import RESULTSDIR
import numpy as np

PLOTDIR = os.path.join(RESULTSDIR, 'prot_vs_teff')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

# logo
all_clusters1 = ['Pleiades', 'Blanco-1', 'Psc-Eri', 'Praesepe', 'NGC-6811']
slow_seq_ages = [450]
model_ids = ['120-Myr', 'Praesepe', 'NGC-6811']
ap.plot_prot_vs_teff(
    outdir, all_clusters1, slow_seq_ages=slow_seq_ages, model_ids=model_ids,
    hide_ax=1, logo_colors=1
)

# figure 1a
all_clusters1 = ['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
                 'Praesepe', 'NGC-6811', 'NGC-6819', 'Ruprecht-147']
slow_seq_ages = list(np.arange(100, 2700, 100))
ap.plot_prot_vs_teff(
    outdir, all_clusters1
)
ap.plot_prot_vs_teff(
    outdir, all_clusters1, slow_seq_ages=slow_seq_ages
)

# figure 1b
all_clusters = ['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
                 'Praesepe', 'NGC-6811']
ap.plot_prot_vs_teff(
    outdir, all_clusters
)
slow_seq_ages = list(np.arange(100, 1100, 100))
ap.plot_prot_vs_teff(
    outdir, all_clusters, slow_seq_ages=slow_seq_ages
)
assert 0

# figure 3 base sketch
slow_seq_ages = [120, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
ap.plot_prot_vs_teff(
    outdir, [], slow_seq_ages=slow_seq_ages
)

##########################################

for poly_order in [2,3,4,5,6,7,8]:
    ap.plot_prot_vs_teff(outdir, all_clusters,
                         model_ids=['120-Myr', '300-Myr', 'Praesepe', 'NGC-6811'],
                         poly_order=poly_order)

#
# single clusters, with reference fits
#
list0 = ['Pleiades', 'Psc-Eri', 'Blanco-1', 'NGC-3532', 'Praesepe', 'NGC-6811']
list1 = ['120-Myr', '120-Myr', '120-Myr', 'NGC-3532', 'Praesepe', 'NGC-6811']
for reference_cluster, model_id in zip(list0, list1):
    for poly_order in [5,6,7]:
        ap.plot_prot_vs_teff(outdir, [reference_cluster], show_binaries=1,
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
    ap.plot_prot_vs_teff(outdir, reference_clusters, show_binaries=1)

# Everything else
do_everything_else = 0
if do_everything_else:
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Praesepe', 'NGC-6811'])
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'Praesepe'])
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'Praesepe', 'NGC-6811'])
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'NGC-3532', 'Praesepe'])
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'Group-X', 'Praesepe'])
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'NGC-3532', 'Praesepe', 'NGC-6811'])
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'NGC-3532', 'Group-X', 'Praesepe', 'NGC-6811'])
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'Blanco-1', 'NGC-3532', 'Group-X', 'Praesepe', 'NGC-6811'])
    ap.plot_prot_vs_teff(outdir, ['Pleiades', 'NGC-3532', 'Group-X', 'NGC-6811'])
