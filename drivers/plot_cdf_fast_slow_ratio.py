import os
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'cdf_fast_slow_ratio')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#

for poly_order in [7]:
    for include_binaries in [1,0]:
        gp.plot_cdf_fast_slow_ratio(
            outdir, poly_order=poly_order,
            model_ids=['α Per', '120-Myr', '300-Myr', 'Praesepe'],
            reference_clusters=['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                                'Group-X', 'Praesepe', 'NGC-6811'],
            include_binaries=include_binaries
        )
        gp.plot_cdf_fast_slow_ratio(
            outdir, poly_order=poly_order,
            model_ids=['120-Myr', '300-Myr', 'Praesepe'],
            reference_clusters=['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                                'Group-X', 'Praesepe', 'NGC-6811'],
            include_binaries=include_binaries
        )
