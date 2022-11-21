import os
import gyroemp.plotting as ap
from gyroemp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'slow_sequence_residual')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#
ap.plot_slow_sequence_residual(
    outdir, ages=[10, 50, 115, 200, 300, 400], bounds_error='limit'
)
ap.plot_slow_sequence_residual(
    outdir, ages=[115, 300, 670, 1000]
)
ap.plot_slow_sequence_residual(
    outdir, ages=[115, 200, 300, 400, 500, 670]
)
