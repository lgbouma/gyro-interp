import pytest
import numpy as np

@pytest.mark.skip(reason="setting up CI (want this to work!)")
def test_binary_checker():

    from gyrointerp.binary_checker import (
        given_source_ids_return_possible_binarity
    )

    # This star is a complex rotator
    source_ids = np.array(['461072508828461824'])
    gaia_datarelease = 'gaiadr3'
    target_df, nbhr_df = given_source_ids_return_possible_binarity(
        source_ids, gaia_datarelease
    )
    assert len(target_df) == 1
    assert len(nbhr_df) == 0

    # This star is a photometric binary and has a high RUWE.  The companion is not
    # resolved by Gaia.  It is also flagged as a non_single_star by the Gaia
    # pipeline.  Here is one possible query for whether it is "possibly a binary".
    source_ids = np.array(['446488105559389568'])
    target_df, nbhr_df = given_source_ids_return_possible_binarity(
        source_ids, gaia_datarelease,
        flag_cutoffs = {
            'rv_error': 10, 'ruwe': 1.3, 'dGmag': 5, 'sep_arcsec': 30
        },
        overwrite=True
    )
    assert len(target_df) == 1
    assert len(nbhr_df) == 1
    assert np.all(target_df.nbhr_count == 1)
    assert np.all(target_df.flag_nbhr_count)

    # Do both stars, going deep for possible neighbors.
    source_ids = np.array(['461072508828461824', '446488105559389568'])
    runid = "test_binary_checker"
    target_df, nbhr_df = given_source_ids_return_possible_binarity(
        source_ids, gaia_datarelease,
        flag_cutoffs = {
            'rv_error': 10, 'ruwe': 1.3, 'dGmag': 5, 'sep_arcsec': 30
        },
        runid=runid
    )

    assert len(target_df) == 2
    assert len(nbhr_df) == 4


if __name__ == "__main__":
    test_binary_checker()
