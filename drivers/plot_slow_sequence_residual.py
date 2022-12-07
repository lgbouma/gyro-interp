import os
import gyrointerp.plotting as ap
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'slow_sequence_residual')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#
ap.plot_slow_sequence_residual(
    outdir, ages=[10, 50, 120, 200, 300, 400], bounds_error='limit'
)
ap.plot_slow_sequence_residual(
    outdir, ages=[120, 300, 670, 1000]
)
ap.plot_slow_sequence_residual(
    outdir, ages=[120, 200, 300, 400, 500, 670]
)
