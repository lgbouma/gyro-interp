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

slow_seq_ages = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
                 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200,
                 2300, 2400, 2500, 2600]

# systematic uncertainty checks
for n in [1.0, 0.2]:
    ap.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median', slow_seq_ages=slow_seq_ages, n=n
    )
    ap.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median_abs', slow_seq_ages=slow_seq_ages, n=n
    )
assert 0

# fig3a (I think...)
ap.plot_empirical_limits_of_gyrochronology(
    outdir, 'both', slow_seq_ages=slow_seq_ages, writepdf=1
)

assert 0

for n in [0.5, 1.0, 0.2]:
    ap.plot_empirical_limits_of_gyrochronology(
        outdir, 'median', slow_seq_ages=slow_seq_ages, n=n
    )
    ap.plot_empirical_limits_of_gyrochronology(
        outdir, 'peak', slow_seq_ages=slow_seq_ages, n=n
    )

assert 0
# figure 3a
ap.plot_empirical_limits_of_gyrochronology(
    outdir, 'both_abs', slow_seq_ages=slow_seq_ages
)
assert 0
# bonus
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
