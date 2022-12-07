import os
from gyrointerp import __path__
__path__ = list(__path__)

DATADIR = os.path.join(os.path.dirname(__path__[0]), 'data')
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')

LOCALDIR = os.path.join(os.path.expanduser('~'), 'local')
if not os.path.exists(LOCALDIR):
    print(f"Making {LOCALDIR}")
    os.mkdir(LOCALDIR)
