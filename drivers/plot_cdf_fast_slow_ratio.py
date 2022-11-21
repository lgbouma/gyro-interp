import os
import gyroemp.plotting as ap
from gyroemp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'cdf_fast_slow_ratio')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#

for poly_order in [7]:
    ap.plot_cdf_fast_slow_ratio(
        outdir, poly_order=poly_order
    )
