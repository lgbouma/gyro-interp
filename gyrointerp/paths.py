"""
This module contains paths that are re-used throughout the project.  Crucially,
it makes a hidden local cache directory, at ~/.gyrointerp_cache, where results
from large batch runs will by default be stored.
"""
import os
from gyrointerp import __path__
__path__ = list(__path__)

DATADIR = os.path.join(__path__[0], 'data')
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
CACHEDIR = os.path.join(os.path.expanduser('~'), '.gyrointerp_cache')

for l in [DATADIR, CACHEDIR]:
    if not os.path.exists(l):
        print(f"Making {l}")
        os.mkdir(l)

# used for making the BPH23 plots, and not the main functionality
LOCALDIR = os.path.join(os.path.expanduser('~'), 'local')
