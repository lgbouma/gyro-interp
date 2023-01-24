import os
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'prot_vs_teff_residual')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#
all_clusters = ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X', 'Praesepe', 'NGC-6811']
model_ids = ['120-Myr', '300-Myr', 'Praesepe', 'NGC-6811']
for poly_order in [7,6,5]:
    gp.plot_prot_vs_teff_residual(
        outdir, all_clusters, model_ids, poly_order=poly_order
    )

all_clusters = ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
                'Praesepe', 'NGC-6819', 'Ruprecht-147']
model_ids = ['120-Myr', '300-Myr', 'Praesepe', '2.6-Gyr']
for poly_order in [7]:
    gp.plot_prot_vs_teff_residual(
        outdir, all_clusters, model_ids, poly_order=poly_order
    )
