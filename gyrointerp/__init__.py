__version__ = 0.1

# the basic logging styles common to all gyro-interp modules
log_sub = '{'
log_fmt = '[{levelname:1.1} {asctime} {module}:{lineno}] {message}'
log_date_fmt = '%y%m%d %H:%M:%S'

from gyrointerp.gyro_posterior import gyro_age_posterior
from gyrointerp.gyro_posterior import gyro_age_posterior_list
from gyrointerp.helpers import get_summary_statistics
