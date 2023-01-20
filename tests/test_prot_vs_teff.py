import pytest
from gyrointerp.plotting import plot_prot_vs_teff

def test_prot_vs_teff():

    # write the results to this directory
    outdir = "./"

    # show these cluster Prot vs Teff datasets
    reference_clusters = [
        'α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
        'Praesepe', 'NGC-6811'
    ]

    # underplot these polynomial fits
    model_ids = [
        'α Per', '120-Myr', '300-Myr', 'Praesepe', 'NGC-6811'
    ]

    # overplot these stars with big markers
    custom_stardict = {
        "Kepler-1643": {"Prot":5.1, "Teff":4916, "m":"s", "c":"red"},
        "TOI-1136": {"Prot":8.7, "Teff":5770, "m":"X", "c":"pink"},
        "TOI-1937 A": {"Prot":6.6, "Teff":5798, "m":"P", "c":"aqua"},
    }

    # make the plot
    plot_prot_vs_teff(
        outdir, reference_clusters=reference_clusters, model_ids=model_ids,
        custom_stardict=custom_stardict, writepdf=0
    )

    assert 1

if __name__ == "__main__":
    test_prot_vs_teff()
