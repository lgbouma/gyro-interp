__version__ = 0.5
__uri__ = "https://gyro-interp.readthedocs.io/"
__author__ = "Luke Bouma"
__email__ = "bouma.luke@gmail.com"
__license__ = "MIT"
__description__ = (
    'Gyrochronology by interpolation of open cluster rotation sequences.'
)

# the basic logging styles common to all gyro-interp modules
log_sub = '{'
log_fmt = '[{levelname:1.1} {asctime} {module}:{lineno}] {message}'
log_date_fmt = '%y%m%d %H:%M:%S'

from gyrointerp.gyro_posterior import gyro_age_posterior
from gyrointerp.gyro_posterior import gyro_age_posterior_list
from gyrointerp.helpers import get_summary_statistics

__all__ = [
    "gyro_age_posterior",
    "gyro_age_posterior_list",
    "get_summary_statistics",
    "__version__",
]
