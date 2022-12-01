import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyroemp.gyro_posterior import gyro_age_posterior
from gyroemp.helpers import given_grid_post_get_summary_statistics

age_grid = np.linspace(0, 2700, 501)
Teff = 5800
Prot = 5.1

age_post = gyro_age_posterior(
    Prot, Teff, age_grid=age_grid, verbose=False
)

r = given_grid_post_get_summary_statistics(age_grid, age_post)
print(r)
import IPython; IPython.embed()
