"""
Internal functions use a pickle file of samples to cache the parameters from
the population hyperparameter sampling.  (e.g.
"fit_120-Myr_300-Myr_Praesepe.pkl").

For external visibility, repackage this as a CSV file with named headers.
"""

import pandas as pd
import pickle
from os.path import join
from gyrointerp.paths import CACHEDIR, LOCALDIR

pklpath = join(LOCALDIR, "gyrointerp", "fitgyro_emcee_v02", "fit_120-Myr_300-Myr_Praesepe.pkl")

with open(pklpath, 'rb') as f:
    d = pickle.load(f)
    flat_samples = d['flat_samples']

# a1, y_g, logk0, logk1, logf
df = pd.DataFrame({
    'a1/a0': flat_samples[:,0],
    'y_g': flat_samples[:,1],
    'log_k0': flat_samples[:,2],
    'log_k1': flat_samples[:,3],
    'log_f': flat_samples[:,4]
})

csvpath = join(LOCALDIR, "gyrointerp", "fitgyro_emcee_v02", "fit_120-Myr_300-Myr_Praesepe.csv")

df.to_csv(csvpath, index=False)
print(f"Wrote {csvpath}")
