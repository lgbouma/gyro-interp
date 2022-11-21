import os
import gyroemp.plotting as ap
from gyroemp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'data_vs_model_prot')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

#
# many clusters, overplotted
#
ap.plot_data_vs_model_prot(outdir)
