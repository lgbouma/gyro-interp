<p align="center"><img src="docs/source/gyrointerp_logo.png" alt="gyrointerp" width="70%"/></p>

[<img src="https://readthedocs.org/projects/gyro-interp/badge/?version=latest">](https://gyro-interp.readthedocs.io/en/latest/index.html) <img src="https://github.com/lgbouma/gyro-interp/workflows/Tests/badge.svg">

## Documentation

The documentation is hosted at
[gyro-interp.readthedocs.io](https://gyro-interp.readthedocs.io/en/latest/index.html).
A minimal example to get you started is below.


## Install
Preferred installation method is through PyPI:
```shell
pip install gyrointerp
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


## Attribution

If you use the code in your work, please reference
```
@ARTICLE{2023arXiv230308830B,
       author = {{Bouma}, Luke G. and {Palumbo}, Elsa K. and {Hillenbrand}, Lynne A.},
        title = "{The Empirical Limits of Gyrochronology}",
      journal = {arXiv e-prints},
         year = 2023,
        month = mar,
          eid = {arXiv:2303.08830},
        pages = {arXiv:2303.08830},
          doi = {10.48550/arXiv.2303.08830},
archivePrefix = {arXiv},
       eprint = {2303.08830},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230308830B}
}
```
This bibtex entry will be updated once the manuscript clears production.

If your result is particularly dependent on the rotation data from any
one cluster, we also encourage you to refer to the relevant study:

* α Per: [Boyle & Bouma (2023)](https://ui.adsabs.harvard.edu/abs/2022arXiv221109822B/abstract)
* Pleiades: [Rebull et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016AJ....152..113R/abstract)
* Blanco-1: [Gillen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.1008G/abstract)
* Psc-Eri: [Curtis et al. (2019a)](https://ui.adsabs.harvard.edu/abs/2019AJ....158...77C/abstract)
* NGC-3532: [Fritzewski et al. (2022)](https://ui.adsabs.harvard.edu/abs/2021A%26A...652A..60F/abstract)
* Group-X: [Messina et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...657L...3M/abstract)
* Praesepe: [Rampalli et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...921..167R/abstract)
* NGC-6811: [Curtis et al. (2019b)](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...49C/abstract)
* NGC-6819: [Meibom et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015Natur.517..589M/abstract)
* Ruprecht-147 [Curtis et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...904..140C/abstract)
* M67: [Barnes et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...823...16B/abstract) and [Dungee et al (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...938..118D/abstract).


