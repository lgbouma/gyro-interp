import os
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'slow_sequence_residual')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#
gp.plot_slow_sequence_residual(
    outdir, ages=[10, 50, 120, 200, 300, 400], bounds_error='limit'
)
gp.plot_slow_sequence_residual(
    outdir, ages=[120, 300, 670, 1000]
)
gp.plot_slow_sequence_residual(
    outdir, ages=[120, 200, 300, 400, 500, 670]
)
