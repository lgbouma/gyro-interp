import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp.gyro_posterior import (
    agethreaded_gyro_age_posterior, gyro_age_posterior
)
from datetime import datetime

bounds_error = 'limit'
age_grid = np.linspace(0, 2700, 1001)
Teff = 5500
Prot = 10

def run_serial(N):

    print(f"{datetime.now().isoformat()} begin N={N} (serial)")
    age_post = gyro_age_posterior(
        Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
        N_grid=N, verbose=False
    )
    print(f"{datetime.now().isoformat()} end N={N} (serial)")
    return age_post

def run_parallel(N):

    print(f"{datetime.now().isoformat()} begin N={N} (parallel)")
    age_post = agethreaded_gyro_age_posterior(
        Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
        N_grid=N, verbose=False
    )
    print(f"{datetime.now().isoformat()} end N={N} (parallel)")
    return age_post

def main():
    age_post = run_serial(256)
    age_post_parallel = run_parallel(256)

    plt.plot(
        age_grid, age_post, label='serial'
    )
    plt.plot(
        age_grid, age_post_parallel, label='parallel'
    )
    plt.legend(loc='best', fontsize='small')

    plt.ylabel('probability')
    plt.xlabel('age [myr]')
    outpath = '../results/age_posteriors/gyro_posterior_testplot_Teff5500_Prot10.png'
    plt.savefig(
        outpath, bbox_inches='tight'
    )
    print(f"wrote {outpath}")
    import IPython; IPython.embed()

if __name__ == "__main__":
    main()
