"""
This test checks that a star that visually appears to overlap with M67
gets an age near 4 Gyr, with a statistical precision between 16-33%.
"""
import os, pytest
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp import gyro_age_posterior
from gyrointerp import get_summary_statistics

def test_gyro_posterior_old():

    #
    # check sun-like star near M67, to ensure extrapolation is working
    #
    age_grid = np.linspace(0, 5000, 501)
    Teff, Teff_err = 5000, 50
    Prot, Prot_err = 31, 3

    age_post = gyro_age_posterior(
        Prot, Teff, Teff_err=Teff_err, Prot_err=Prot_err, age_grid=age_grid,
        verbose=False, bounds_error='4gyrextrap'
    )

    r = get_summary_statistics(age_grid, age_post)

    assert abs(r['median'] - 4000) < 200
    assert abs(r['mean'] - 4000) < 200
    assert 200 < r['+1sigma'] < 700
    assert 200 < r['-1sigma'] < 700

    print(r)


if __name__ == "__main__":
    test_gyro_posterior_old()
