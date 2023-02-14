import os
from glob import glob
import pandas as pd, numpy as np

csvpaths = glob("/Users/luke/Dropbox/proj/gyro-interp/results/prot_vs_teff/polyorder*csv")
df = pd.concat((pd.read_csv(f) for f in csvpaths))

df['age'] = 0
df.loc[df.model_id=='120-Myr', 'age'] = 120
df.loc[df.model_id=='300-Myr', 'age'] = 300
df.loc[df.model_id=='Praesepe', 'age'] = 670
df.loc[df.model_id=='NGC-6811', 'age'] = 1000

df = df.sort_values(by=['age', 'k'])

cols = 'chi_sq_red'.split(",")
for col in cols:
    df[col] = np.round(df[col], 2)
cols = 'chi_sq,AIC,BIC'.split(",")
for col in cols:
    df[col] = np.round(df[col], 1)

cols = ['model_id', 'Nstar', 'k', 'chi_sq', 'chi_sq_red', 'AIC', 'BIC']
print(df[cols])
