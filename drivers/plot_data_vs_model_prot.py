import os
import gyroemp.plotting as ap
from gyroemp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'data_vs_model_prot')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#
ap.plot_data_vs_model_prot(
    outdir,
    model_ids=['α Per', '120-Myr', '300-Myr', 'Praesepe'],
    reference_clusters=['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                        'Group-X', 'Praesepe', 'NGC-6811']
)
assert 0

ap.plot_data_vs_model_prot(
    outdir,
    model_ids=['α Per', '120-Myr', '300-Myr'],
    reference_clusters=['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                        'Group-X', 'Praesepe', 'NGC-6811']
)

ap.plot_data_vs_model_prot(
    outdir,
    model_ids=['120-Myr', '300-Myr', 'Praesepe'],
    reference_clusters=['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                        'Group-X', 'Praesepe', 'NGC-6811']
)


