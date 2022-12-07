import pytest
import numpy as np
from gyrointerp.models import slow_sequence

def test_slow_sequence():
    teff_arr = np.linspace(3800, 6200, 2)
    prot_arr = slow_sequence(teff_arr, 115)
    assert isinstance(prot_arr, np.ndarray) and prot_arr.shape == (2,)

    prot_arr = slow_sequence(5000, 130)
    assert isinstance(prot_arr, np.ndarray)
    assert (prot_arr[0] > 1) & (prot_arr[0] < 10)

    prot_arr = slow_sequence(5000, 200)
    assert isinstance(prot_arr, np.ndarray)
    assert (prot_arr[0] > 1) & (prot_arr[0] < 10)

    prot_arr = slow_sequence(7000, 200)
    assert np.isnan(prot_arr).sum() == 1

    prot_arr = slow_sequence(5000, 115)
    assert isinstance(prot_arr, np.ndarray)
    assert (prot_arr[0] > 1) & (prot_arr[0] < 10)

    teff_arr = np.linspace(3800, 6200, 100)
    prot_arr = slow_sequence(teff_arr, 150)
    assert isinstance(prot_arr, np.ndarray)

if __name__ == "__main__":
    test_slow_sequence()
