"""
Concatenate the results from parallel_fit_gyro_model.py
Write the output to one CSV file.
(which will be plotted by plot_fit_gyro_model.py)
"""
from itertools import product
from gyroemp.fitting import get_chi_sq_red
from gyroemp.paths import RESULTSDIR, LOCALDIR
import os
from glob import glob
import numpy as np, pandas as pd
from datetime import datetime

modelid = "fitgyro_v02_zeroB_zeroA_N40k"
input_dir = os.path.join(LOCALDIR, "gyroemp", modelid)

csvpaths = glob(os.path.join(input_dir, "A_*csv"))
N = len(csvpaths)
print(N)

outdir = os.path.join(RESULTSDIR, "fit_gyro_model")
outpath = os.path.join(outdir, f'{modelid}_concatenated_chi_squared_results.csv')

for ix, csvpath in enumerate(csvpaths):

    # Funky file handling just to avoid stale pointers
    if ix % 10000 == 0:
        now = datetime.now().isoformat()
        print(f"{now}: {ix}/{N}...")
        if ix == 0:
            write_handle = open(outpath, "a")
        else:
            write_handle.close()
            write_handle = open(outpath, "a")

    with open(csvpath, "r") as f:
        in_lines = f.readlines()

    if len(in_lines) == 2:
        write_handle.write(in_lines[-1])

    else:
        print(f"ERR! {csvpath} did not have exactly two lines")
        pass

write_handle.close()
print(f"Wrote {outpath}")
