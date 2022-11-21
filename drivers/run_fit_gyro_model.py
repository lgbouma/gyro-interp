"""
Varying A, B, C, k0, and k2, each over a grid, what parameters provide the best
fit to the data?
"""

from itertools import product
from gyroemp.fitting import get_chi_sq_red
from gyroemp.paths import RESULTSDIR, LOCALDIR
import os
from glob import glob
import numpy as np, pandas as pd

outdir = os.path.join(LOCALDIR, "young-KOIs")
if not os.path.exists(outdir): os.mkdir(outdir)

A_grid = np.arange(1.0, 2.1, 0.3)
B_grid = np.arange(1.0, 4.6, 0.3)
C_grid = np.arange(0.5, 2.1, 0.3)
C_y0_grid = np.arange(0.2, 0.9, 0.2)
#k0_grid = [np.e**-6, np.e**-5, np.e**-4]
#k2_grid = [np.e**-8, np.e**-7.5, np.e**-7, np.e**-6.5]
logk0_grid = np.arange(-6,-3.9,0.3)
logk2_grid = np.arange(-9,-6.4,0.3)

N = (
    len(A_grid) * len(B_grid) * len(C_grid) * len(C_y0_grid)
    * len(logk0_grid) *len(logk2_grid)
)
print(N)

sigma_period = 0.51
l1 = -2*sigma_period
k1 = np.e**1

for A,B,C,C_y0,logk0,logk2 in product(
    A_grid, B_grid, C_grid, C_y0_grid, logk0_grid, logk2_grid
):

    parameters = {
        'A': A,
        'B': B,
        'C': C,
        'C_y0': C_y0,
        'k0': np.exp(logk0),
        'k1': k1,
        'l1': l1,
        'k2': np.exp(logk2)
    }

    outname = (
        f"A_{A:.1f}_B_{B:.1f}_C_{C:.1f}_Cy0_{C_y0:.2f}_logk0_{logk0:.2f}_"
        f"logk1_{np.log(k1):.2f}_l1_{l1:.5f}_logk2_{logk2:.2f}.csv"
    )
    cachepath = os.path.join(outdir, outname)

    if not os.path.exists(cachepath):
        chi_sq_red = get_chi_sq_red(parameters, cachepath)
        outdf = pd.DataFrame(parameters,index=[0])
        outdf['chi_sq_red'] = chi_sq_red
        outdf.to_csv(cachepath, index=False)
        print(f"Wrote {cachepath}")
    else:
        print(f"Found {cachepath}")


csvpaths = glob(os.path.join(outdir, "A_*csv"))

df = pd.concat((pd.read_csv(f) for f in csvpaths))

df['logk0'] = np.log(df.k0)
df['logk1'] = np.log(df.k1)
df['logk2'] = np.log(df.k2)
df = df.drop(['k0','k1','k2'], axis='columns')

print(df.sort_values(by='chi_sq_red').head(n=100))

import IPython; IPython.embed()
