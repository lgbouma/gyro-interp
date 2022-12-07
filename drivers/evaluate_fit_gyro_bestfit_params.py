"""
DEPRECATED (uses the chi^2 brute-force fitting rather than sampling)
"""
import numpy as np, pandas as pd
from gyrointerp.paths import RESULTSDIR
from datetime import datetime
import os

modelid = "fitgyro_v06_zeroB_zeroA_N750k"

csvpath = os.path.join(
    RESULTSDIR, "fit_gyro_model",
    f"{modelid}_concatenated_chi_squared_results.csv"
)

df = pd.read_csv(
    csvpath, names=f"A,B,C,C_y0,logk0,k1,l1,logk2,chi_sq_red,BIC,n,k".split(',')
)
df['logk1'] = np.log(df.k1)
df = df.drop(['k1'], axis='columns')
df['n'] = 21
df['k'] = 5

df['chi_sq'] = (df.chi_sq_red) * (df.n - df.k)
df['w'] = (
    np.exp(-0.5 * df.chi_sq)
    /
    np.sum(np.exp(-0.5 * df.chi_sq))
)

print(df.sort_values(by='chi_sq_red').head(n=20))

np.random.seed(42)
N=int(1e6)
print(f'{datetime.utcnow().isoformat()}: beginning sample {N}...')
sdf = df.sample(
    n=N,
    replace=True,
    weights=df.w
)
print(f'{datetime.utcnow().isoformat()}: end sample {N}')

selcols = ['C','C_y0','logk0','logk2']

print(42*'-')
print('fit gyro results')
stat_df = sdf[selcols].describe(percentiles=(0.16,0.5,0.84))
print(42*'-')

_df = stat_df.T
_df['+1sigma'] = _df['84%'] - _df['50%']
_df['-1sigma'] = _df['50%'] - _df['16%']

stat_df = _df.T
print(stat_df)
