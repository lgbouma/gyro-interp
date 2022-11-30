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

slow_seq_ages = np.arange(100, 2600+100, 100)

# systematic uncertainty checks for varying age scale
for age_scale in ["1sigmaolder", "1sigmayounger"]:
    ap.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median', slow_seq_ages=slow_seq_ages,
        age_scale=age_scale
    )
    ap.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median_abs', slow_seq_ages=slow_seq_ages,
        age_scale=age_scale
    )

assert 0

# fig3a
ap.plot_empirical_limits_of_gyrochronology(
    outdir, 'both', slow_seq_ages=slow_seq_ages, writepdf=1
)


# systematic uncertainty checks for varying n
for n in [1.0, 0.2]:
    ap.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median', slow_seq_ages=slow_seq_ages, n=n
    )
    ap.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median_abs', slow_seq_ages=slow_seq_ages, n=n
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
