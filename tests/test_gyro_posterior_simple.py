"""
This test checks that a star that visually appears to overlap with NGC-3532
gets an age near 300 Myr, with a statistical precision between 16-33%.
"""
import os, pytest
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp.gyro_posterior import gyro_age_posterior
from gyrointerp.helpers import given_grid_post_get_summary_statistics

def test_gyro_posterior_simple():

    age_grid = np.linspace(0, 2700, 501)
    Teff = 5800
    Prot = 5.1

    age_post = gyro_age_posterior(
        Prot, Teff, age_grid=age_grid, verbose=False
    )

    r = given_grid_post_get_summary_statistics(age_grid, age_post)

    assert abs(r['median'] - 300) < 20
    assert abs(r['peak'] - 300) < 20
    assert abs(r['mean'] - 300) < 20
    assert 50 < r['+1sigma'] < 100
    assert 50 < r['-1sigma'] < 100

if __name__ == "__main__":
    test_gyro_posterior_simple()
