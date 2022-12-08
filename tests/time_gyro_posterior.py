"""
When run on a standard MacBook, this test shows:

401 ages, N=256 grid:
    * gyro_age_posterior takes 4 seconds.
    * _agethreaded_gyro_age_posterior takes 4 seconds.

401 ages, N=512 grid:
    * gyro_age_posterior takes 10 seconds.
    * _agethreaded_gyro_age_posterior takes 6 seconds.

1001 ages, N=256 grid:
    * gyro_age_posterior takes 10 seconds.
    * _agethreaded_gyro_age_posterior takes 5 seconds.

where the macbook has 16 cores to use during _agethreaded_gyro_age_posterior.
"""
import os, pytest
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp.gyro_posterior import (
    _agethreaded_gyro_age_posterior, gyro_age_posterior
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
    age_post = _agethreaded_gyro_age_posterior(
        Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
        N_grid=N, verbose=False
    )
    print(f"{datetime.now().isoformat()} end N={N} (parallel)")
    return age_post


if __name__ == "__main__":
    run_serial(256)
    run_parallel(256)
