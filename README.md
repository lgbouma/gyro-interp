# gyro-interp

<img src="https://github.com/lgbouma/gyro-interp/workflows/Tests/badge.svg">

## Installation
Preferred:
`$ pip install gyrointerp`

Or clone the repository, and:
`$ python setup.py develop`

## Usage

Given a star's rotation period, effective temperature, and uncertainties, what
is the gyrochronological age posterior over a grid spanning 0 to 2.6 Gyr?

```
import numpy as np
from gyroemp import gyro_age_posterior
Prot, Prot_err = 6, 0.05
Teff, Teff_err = 5500, 100
age_grid = np.linspace(0, 2600, 100)
age_posterior = gyro_age_posterior(
  Prot, Teff, Prot_err=Prot_err, Teff_err=Teff_err, age_grid=age_grid
)
```

Auxiliary tool: is my star potentially a binary?  If is, you should be
concerned about that affecting the rotation period, due to a story involving
disk dispersal that's been described by eg.,
ui.adsabs.harvard.edu/abs/2007ApJ...665L.155M/, and has been borne out by more
data since then.

```
from gyroemp.binary_checker import (
    given_source_ids_return_possible_binarity
)

# given a Gaia DR2 or DR3 source_id, determine whether the star is potentially
# binary

source_id = '446488105559389568'
gaia_datarelease = 'gaiadr3'
target_df, nbhr_df = given_source_id_return_possible_binarity(
  source_id, gaia_datarelease
)
```
