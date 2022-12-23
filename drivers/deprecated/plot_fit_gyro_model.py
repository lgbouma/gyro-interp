import os
import gyrointerp.plotting as ap
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'fit_gyro_model')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#

modelid = "fitgyro_v06_zeroB_zeroA_N750k"

ap.plot_fit_gyro_model(outdir, modelid)
