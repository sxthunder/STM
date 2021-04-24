from stm.core import StatisticTextMatching
from stm import preprocess
from stm import utils
from stm import match
from stm import similarity

__author__ = 'Liang Ming'
__all__ = ['StatisticTextMatching', 'preprocess', 'utils', 'match', 'similarity']
__version__ = '1.0.0'
__version_info__ = tuple(__version__.split('.'))

get_version = lambda: __version_info__