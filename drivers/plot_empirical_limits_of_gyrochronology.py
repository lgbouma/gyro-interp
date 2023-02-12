import os
import numpy as np
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'empirical_limits_of_gyrochronology')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR


slow_seq_ages = np.arange(100, 2600+100, 100)

# fig4, publication-quality
gp.plot_empirical_limits_of_gyrochronology(
    outdir, 'both', slow_seq_ages=slow_seq_ages, writepdf=1,
    grid_resolution='fine'
)
assert 0

# fig4, coarse
gp.plot_empirical_limits_of_gyrochronology(
    outdir, 'both', slow_seq_ages=slow_seq_ages, writepdf=1,
    grid_resolution='coarse'
)
assert 0

# systematic uncertaity checks for varying spin-down rates, as assessed through
# interpolation methods ("n")
for interp_method in ["1d_linear", "1d_pchip"]:
    gp.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median', slow_seq_ages=slow_seq_ages,
        interp_method=interp_method
    )
    gp.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median_abs', slow_seq_ages=slow_seq_ages,
        interp_method=interp_method
    )
assert 0

# systematic uncertainty checks for varying age scale
for age_scale in ["1sigmaolder", "1sigmayounger"]:
    gp.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median', slow_seq_ages=slow_seq_ages,
        age_scale=age_scale
    )

# (deprecated) systematic uncertainty checks for varying n
for n in [1.0, 0.2]:
    gp.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median', slow_seq_ages=slow_seq_ages, n=n
    )
    gp.plot_empirical_limits_of_gyrochronology(
        outdir, 'diff_median_abs', slow_seq_ages=slow_seq_ages, n=n
    )

assert 0
# (deprecated) figure 3a, ina absolute values)
gp.plot_empirical_limits_of_gyrochronology(
    outdir, 'both_abs', slow_seq_ages=slow_seq_ages
)

assert 0
# bonus
gp.plot_empirical_limits_of_gyrochronology(
    outdir, 'plus_abs', slow_seq_ages=slow_seq_ages
)
gp.plot_empirical_limits_of_gyrochronology(
    outdir, 'minus_abs', slow_seq_ages=slow_seq_ages
)
gp.plot_empirical_limits_of_gyrochronology(
    outdir, 'plus', slow_seq_ages=slow_seq_ages
)
gp.plot_empirical_limits_of_gyrochronology(
    outdir, 'minus', slow_seq_ages=slow_seq_ages
)
