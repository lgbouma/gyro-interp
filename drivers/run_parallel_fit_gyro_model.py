"""
Varying A, B, C, k0, and k2, each over a grid, what parameters provide the best
fit to the data?

Usage:
    Check modelid and grid matches what you want.  Then:
    $ python -u run_parallel_fit_gyro_model &> logs/logname.log &
"""

from itertools import product
from gyrointerp.fitting import get_chi_sq_red
from gyrointerp.paths import RESULTSDIR, LOCALDIR
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

    modelid = "fitgyro_v07_zeroB_zeroA_N1pt5M"
    outdir = os.path.join(LOCALDIR, "gyrointerp", modelid)
    if not os.path.exists(outdir): os.mkdir(outdir)

    A_grid = np.array([1])
    if is_logB:
        logB_grid = np.arange(-6, 2.0, 1.0)
    else:
        B_grid = np.array([0])
    C_grid = np.arange(1.1, 20.1, 0.2)
    C_y0_grid = np.arange(0.4, 0.8, 0.01)
    logk0_grid = np.arange(-8, 3, 0.5)
    logk2_grid = np.arange(-8,-4.5,0.2)
    # NOTE: probably use these for the 2M resolution
    # C_grid = np.arange(1.1, 5.1, 0.1)
    # C_y0_grid = np.arange(0.2, 0.9, 0.01)
    # logk0_grid = np.arange(-6,-3.9,0.1)
    # logk2_grid = np.arange(-8,-4.5,0.1)

    sigma_period = 0.51
    l1 = -2*sigma_period
    k1 = np.pi

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
    #assert 0

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
