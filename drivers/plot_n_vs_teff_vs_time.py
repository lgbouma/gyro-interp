import os
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'n_vs_teff_vs_time')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

gp.plot_n_vs_teff_vs_time(PLOTDIR)

