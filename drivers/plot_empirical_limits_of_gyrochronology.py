import os
import gyroemp.plotting as ap
from gyroemp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'empirical_limits_of_gyrochronology')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#

slow_seq_ages = [120, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
                 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200,
                 2300, 2400, 2500, 2600]

ap.plot_empirical_limits_of_gyrochronology(
    outdir, 'plus_abs', slow_seq_ages=slow_seq_ages
)
ap.plot_empirical_limits_of_gyrochronology(
    outdir, 'minus_abs', slow_seq_ages=slow_seq_ages
)
ap.plot_empirical_limits_of_gyrochronology(
    outdir, 'plus', slow_seq_ages=slow_seq_ages
)
ap.plot_empirical_limits_of_gyrochronology(
    outdir, 'minus', slow_seq_ages=slow_seq_ages
)
