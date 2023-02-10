import os
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'data_vs_model_prot')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#
for include_binaries in [0,1]:
    gp.plot_data_vs_model_prot(
        outdir,
        model_ids=['α Per', '120-Myr', '300-Myr', 'Praesepe'],
        reference_clusters=['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                            'Group-X', 'Praesepe', 'NGC-6811'],
        include_binaries=include_binaries
    )
assert 0

gp.plot_data_vs_model_prot(
    outdir,
    model_ids=['α Per', '120-Myr', '300-Myr'],
    reference_clusters=['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                        'Group-X', 'Praesepe', 'NGC-6811']
)

gp.plot_data_vs_model_prot(
    outdir,
    model_ids=['120-Myr', '300-Myr', 'Praesepe'],
    reference_clusters=['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                        'Group-X', 'Praesepe', 'NGC-6811']
)


