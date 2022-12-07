import pytest
import numpy as np
from gyrointerp.gaia import given_source_ids_get_gaia_data

@pytest.mark.skip(reason="setting up CI (& shouldnt this be in cdips?)")
def test_kepler1627_allcols():
    # Get all the Gaia columns for Kepler-1627
    source_ids = [np.int64(2103737241426734336)]
    cache_string = 'kepler_1627_test_allcols'
    df = given_source_ids_get_gaia_data(
        source_ids, cache_string, which_columns='*',
        gaia_datarelease='gaiadr2'
    )

    assert len(df) == 1, 'Expected 1 results for Kepler-1627'
    assert len(df.T) == 96, 'Expected 96 columns for Kepler-1627'


@pytest.mark.skip(reason="setting up CI (& shouldnt this be in cdips?)")
def test_kepler1627_selectcols():
    # Get all the Gaia columns for Kepler-1627
    source_ids = [np.int64(2103737241426734336)]
    cache_string = 'kepler_1627_test_selectcols'
    df = given_source_ids_get_gaia_data(
        source_ids, cache_string,
        which_columns='ra, dec, pmra, pmdec',
        gaia_datarelease='gaiadr2', overwrite=True
    )

    assert len(df) == 1
    assert len(df.T) == 4, 'Expected 4 columns: ra,dec,pmra,pmdec'


if __name__ == "__main__":
    test_kepler1627_selectcols()
    test_kepler1627_allcols()
