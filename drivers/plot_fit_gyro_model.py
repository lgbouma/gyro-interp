import os
import gyroemp.plotting as ap
from gyroemp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'fit_gyro_model')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#

modelid = "fitgyro_v5_gridlogB"
modelid = "fitgyro_v6_zeroB"
modelid = "fitgyro_v7_zeroB_denser"

ap.plot_fit_gyro_model(outdir, modelid)
