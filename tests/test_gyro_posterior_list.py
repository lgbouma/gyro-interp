import pytest

import numpy as np, pandas as pd
from gyrointerp import gyro_age_posterior_list, get_summary_statistics

@pytest.mark.skip(reason="setting up CI")
def test_gyro_posterior_list():

    N_stars = 8
    Teffs = np.linspace(4000, 5500, N_stars)
    Teff_errs = 100 *np.ones(N_stars)
    # at >~20 days, assume a few percent relative uncertainty on periods
    Prots = np.linspace(15, 22, N_stars)
    Prot_errs = 0.03 * Prots

    # The output posteriors will be cached at ~/.gyrointerp_cache/{cache_id}
    cache_id = 'my_awesome_stars'

    # That 5500 K star with Prot = 22 days is near the Ruprecht-147 sequence.
    # Let's extend the age_grid up to 4000 Myr (4 Gyr), understanding that the
    # extrapolation past 2.6 Gyr is based on the M67 data.
    age_grid = np.linspace(0, 4000, 500)

    # Let's pass optional star IDs to name the posteriors in an obvious way.
    star_ids = [f"STAR{ix}" for ix in range(N_stars)]

    csvpaths = gyro_age_posterior_list(
        cache_id, Prots, Teffs, Prot_errs=Prot_errs, Teff_errs=Teff_errs,
        star_ids=star_ids, age_grid=age_grid, bounds_error="4gyrlimit",
        interp_method="pchip_m67"
    )

    # Let's read the posteriors and print their summary statistics.
    for csvpath, Prot, Teff in zip(sorted(csvpaths), Prots, Teffs):
        df = pd.read_csv(csvpath)
        r = get_summary_statistics(df.age_grid, df.age_post)
        msg = f"Age = {r['median']} +{r['+1sigma']} -{r['-1sigma']} Myr."
        print(f"Teff {int(Teff)} Prot {Prot:.2f} {msg}")


if __name__ == "__main__":
    test_gyro_posterior_list()
