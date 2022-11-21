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
import multiprocessing as mp

def chi_sq_worker(task):

    #_B is either "B" or "logB"
    outdir, A, _B, C, C_y0, logk0, logk2, l1, k1, is_logB = task

    b_key = 'logB' if is_logB else 'B'
    parameters = {
        'A': A,
        b_key: _B,
        'C': C,
        'C_y0': C_y0,
        'logk0': logk0,
        'k1': k1,
        'l1': l1,
        'logk2': logk2
    }

    Bstr = 'logB' if is_logB else 'B'

    outname = (
        f"A_{A:.1f}_{Bstr}_{_B:.1f}_C_{C:.1f}_Cy0_{C_y0:.2f}_logk0_{logk0:.2f}_"
        f"logk1_{np.log(k1):.2f}_l1_{l1:.5f}_logk2_{logk2:.2f}.csv"
    )

    cachepath = os.path.join(outdir, outname)

    if not os.path.exists(cachepath):
        chi_sq_red, BIC = get_chi_sq_red(parameters, cachepath)
        outdf = pd.DataFrame(parameters,index=[0])
        outdf['chi_sq_red'] = chi_sq_red
        outdf['BIC'] = BIC
        outdf.to_csv(cachepath, index=False)
        print(f"Wrote {cachepath}")
    else:
        print(f"Found {cachepath}")


def main():

    is_logB = False # set this

    modelid = "fitgyro_v7_zeroB_denser"
    outdir = os.path.join(LOCALDIR, "young-KOIs", modelid)
    if not os.path.exists(outdir): os.mkdir(outdir)

    A_grid = np.arange(0.1, 2.1, 0.2)
    if is_logB:
        logB_grid = np.arange(-6, 2.0, 1.0)
    else:
        B_grid = [0]
    C_grid = np.arange(0.5, 2.1, 0.2)
    C_y0_grid = np.arange(0.2, 0.9, 0.1)
    logk0_grid = np.arange(-6,-3.9,0.5)
    logk2_grid = np.arange(-8,-3.9,0.5)

    sigma_period = 0.51
    l1 = -2*sigma_period
    k1 = np.e**1

    if is_logB:
        N = (
            len(A_grid) * len(logB_grid) * len(C_grid) * len(C_y0_grid)
            * len(logk0_grid) *len(logk2_grid)
        )
        tasks = [
            (outdir, A, logB, C, C_y0, logk0, logk2, l1, k1, is_logB) for
            A, logB, C, C_y0, logk0,logk2 in product(
                A_grid, logB_grid, C_grid, C_y0_grid, logk0_grid, logk2_grid
            )
        ]
    else:
        N = (
            len(A_grid) * len(B_grid) * len(C_grid) * len(C_y0_grid)
            * len(logk0_grid) *len(logk2_grid)
        )
        tasks = [
            (outdir, A, B, C, C_y0, logk0, logk2, l1, k1, is_logB) for
            A, B, C, C_y0, logk0,logk2 in product(
                A_grid, B_grid, C_grid, C_y0_grid, logk0_grid, logk2_grid
            )
        ]

    print(N)

    nworkers = mp.cpu_count()
    maxworkertasks = 1000
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(chi_sq_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
