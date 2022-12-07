"""
Contents:

Main re-useable functions:
    given_source_ids_get_gaia_data

Helper functions:
    _make_votable_given_source_ids
    _given_votable_get_df
"""

###########
# imports #
###########
import os
import numpy as np, matplotlib.pyplot as plt, pandas as pd

from astropy.io.votable import from_table, writeto, parse
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from astroquery.gaia import Gaia

##########
# config #
##########

homedir = os.path.expanduser("~")
credentials_file = os.path.join(homedir, '.gaia_credentials')
if not os.path.exists(credentials_file):
    raise AssertionError(
        f'Need to make an account at https://gea.esac.esa.int/archive/ '
        'and then create a Gaia credentials file at {credentials_file}. '
        'See https://astroquery.readthedocs.io/en/latest/gaia/gaia.html#login-logout'
    )

gaiadir = os.path.join(homedir, '.gaia_cache')
if not os.path.exists(gaiadir):
    print(f'Making {gaiadir} to cache Gaia query results.')
    os.mkdir(gaiadir)

#############
# functions #
#############

def given_source_ids_get_gaia_data(
    source_ids, cache_string, n_max=10000, overwrite=False,
    enforce_all_sourceids_viable=True, which_columns='*',
    gaia_datarelease='gaiadr2'
):
    """
    Args:

        source_ids (np.ndarray): list or array of np.int64 Gaia DR2 or EDR3
        source_ids.  If EDR3, be sure to use the correct `gaia_datarelease`
        kwarg.

        cache_string (str): unique identifier string used to cache the list of
        source identifiers.

        n_max (int): maximum number of sources to run in the SQL query.

        overwrite (bool): if True, and finds that this crossmatch has already
        run, deletes previous cached output and reruns anyway.

        enforce_all_sourceids_viable (bool): if True, will raise an assertion
        error if every requested source_id does not return a result. (Unless
        the query returns n_max entries, in which case only a warning will be
        raised).

        which_columns (str): ADQL column getter string. Examples include "*"
        (to get all possible columns), or "ra, dec, pm, pmdec" to select only
        those columns.  Defaults to all possible columns.

        gaia_datarelease (str): 'gaiadr2' or 'gaiaedr3'. Default is Gaia DR2.

    Returns:

        pandas dataframe with Gaia DR2 / EDR3 crossmatch info.
    """

    #
    # Check the inputs
    #
    assert gaia_datarelease in ['gaiadr2', 'gaiaedr3']

    if n_max > int(5e4):
        raise NotImplementedError(
            'The gaia archive / astroquery seems to give invalid results past '
            '50000 source_ids in this implementation...'
        )

    if type(source_ids) not in [np.ndarray, list]:
        raise TypeError(
            'source_ids must be np.ndarray or list of np.int64 '
            'Gaia DR2 source_ids'
        )
    if type(source_ids[0]) != np.int64:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 '
            'Gaia DR2 source_ids'
        )

    #
    # Convert the inputs into a cached votable that will be uploaded to the
    # Gaia archive's TAP portal to perform the crossmatch.
    #
    xml_to_upload_path = os.path.join(
        gaiadir, f'toupload_{cache_string}_{gaia_datarelease}.xml'
    )
    gaia_download_path = os.path.join(
        gaiadir, f'group{cache_string}_matches_{gaia_datarelease}.xml.gz'
    )

    if overwrite:
        if os.path.exists(xml_to_upload_path):
            os.remove(xml_to_upload_path)

    if not os.path.exists(xml_to_upload_path):
        _make_votable_given_source_ids(source_ids, outpath=xml_to_upload_path)

    if os.path.exists(gaia_download_path) and overwrite:
        print(f'WRN! Removing {gaia_download_path}, since overwrite is True.')
        os.remove(gaia_download_path)

    #
    # Define the ADQL job and then run it.  Cache the results.
    #
    jobstr = (
    """
    SELECT top {n_max:d} {which_columns}
    FROM tap_upload.foobar as u, {gaia_datarelease:s}.gaia_source AS g
    WHERE u.source_id=g.source_id
    """
    ).format(
        n_max=n_max,
        which_columns=which_columns,
        gaia_datarelease=gaia_datarelease
    )

    if not os.path.exists(gaia_download_path):

        Gaia.login(credentials_file=credentials_file)

        j = Gaia.launch_job(
            query=jobstr, upload_resource=xml_to_upload_path,
            upload_table_name="foobar", verbose=True, dump_to_file=True,
            output_file=gaia_download_path
        )

        Gaia.logout()

    df = _given_votable_get_df(gaia_download_path)

    #
    # If some of the Gaia source IDs that were passed did not return useful
    # results, we want to raise warnings or errors, depending on whether
    # `enforce_all_sourceids_viable` was set to be True.
    #
    if len(df) != len(source_ids) and enforce_all_sourceids_viable:
        if len(df) == n_max:
            wrnmsg = (
                f'WRN! got {len(df)} matches vs {len(source_ids)} '
                'source id queries'
            )
            print(wrnmsg)
        else:
            errmsg = (
                f'ERROR! got {len(df)} matches vs {len(source_ids)} '
                'source id queries'
            )
            raise AssertionError(errmsg)

    if len(df) != len(source_ids) and not enforce_all_sourceids_viable:
        wrnmsg = (
            f'WRN! got {len(df)} matches vs {len(source_ids)} '
            'source id queries'
        )

    return df


def _given_votable_get_df(votablepath):
    # helper function to convert between a votable stored on the hard-drive to
    # a pandas dataframe in local memory.

    vot = parse(votablepath)
    tab = vot.get_first_table().to_table()
    df = tab.to_pandas()

    return df


def _make_votable_given_source_ids(source_ids, outpath=None):
    # helper function to make a votable of source identifiers to upload to the
    # gaia archive.  used in given_source_ids_get_gaia_data.

    t = Table()
    t['source_id'] = source_ids

    votable = from_table(t)

    writeto(votable, outpath)
    print(f'Made {outpath}')

    return outpath
