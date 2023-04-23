import logging

from . import config

logger = logging.getLogger("scrapscope")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.getLevelNamesMapping()[config.log_level])
