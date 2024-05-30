"""
This test checks that stars that visually overlap with NGC-3532 and
Ruprecht-147 get ages near 300 Myr and 2.6 Gyr, with a statistical precision
between 16-33%.
"""
import os, pytest
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp.gyro_posterior import gyro_age_posterior
from gyrointerp.helpers import get_summary_statistics

def test_gyro_posterior_simple():

    #
    # check sun-like star near NGC-3532
    # NOTE: this is also a regression test from a case mentioned in the ms
    #
    age_grid = np.linspace(0, 2700, 501)
    Teff, Teff_err = 5800, 50
    Prot = 5.1

    age_post = gyro_age_posterior(
        Prot, Teff, Teff_err=Teff_err, age_grid=age_grid, verbose=False
    )

    r = get_summary_statistics(age_grid, age_post)

    assert abs(r['median'] - 300) < 20
    assert abs(r['peak'] - 300) < 30
    assert abs(r['mean'] - 300) < 20
    assert 50 < r['+1sigma'] < 100
    assert 50 < r['-1sigma'] < 100

    #
    # check sun-like star near Ruprecht-147, to ensure extrapolation is working
    #
    age_grid = np.linspace(0, 5000, 501)
    Teff, Teff_err = 5800, 50
    Prot = 17

    age_post = gyro_age_posterior(
        Prot, Teff, Teff_err=Teff_err, age_grid=age_grid, verbose=False
    )

    r = get_summary_statistics(age_grid, age_post)

    assert abs(r['median'] - 2600) < 200
    assert abs(r['mean'] - 2600) < 200
    assert 200 < r['+1sigma'] < 300
    assert 200 < r['-1sigma'] < 300


if __name__ == "__main__":
    test_gyro_posterior_simple()
