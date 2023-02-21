<p align="center"><img src="docs/source/gyrointerp_logo.png" alt="gyrointerp" width="70%"/></p>

[<img src="https://readthedocs.org/projects/gyro-interp/badge/?version=latest">](https://gyro-interp.readthedocs.io/en/latest/index.html) <img src="https://github.com/lgbouma/gyro-interp/workflows/Tests/badge.svg">

## Documentation

The documentation is hosted at
[gyro-interp.readthedocs.io](https://gyro-interp.readthedocs.io/en/latest/index.html).
A minimal example to get you started is below.


## Install
Preferred installation method:
```shell
pip install gyrointerp
```

Or 
```shell
git clone https://github.com/lgbouma/gyro-interp
cd gyro-interp
python setup.py install
```

## Minimal Example
Given a single star's rotation period, effective temperature, and
uncertainties, what is the gyrochronological age posterior over a grid spanning
0 to 2.6 Gyr?

```python
  import numpy as np
  from gyrointerp import gyro_age_posterior

  # units: days
  Prot, Prot_err = 11, 0.2

  # units: kelvin
  Teff, Teff_err = 4500, 100

  # uniformly spaced grid between 0 and 2600 megayears
  age_grid = np.linspace(0, 2600, 500)

  # calculate the age posterior - takes ~30 seconds
  age_posterior = gyro_age_posterior(
      Prot, Teff, Prot_err=Prot_err, Teff_err=Teff_err, age_grid=age_grid
  )

  # calculate dictionary of summary statistics
  from gyrointerp import get_summary_statistics
  result = get_summary_statistics(age_grid, age_posterior)

  print(f"Age = {result['median']} +{result['+1sigma']} -{result['-1sigma']} Myr.")
```

[The documentation](https://gyro-interp.readthedocs.io/en/latest/index.html)
contains more extensive examples, as well as discussion of important caveats.
