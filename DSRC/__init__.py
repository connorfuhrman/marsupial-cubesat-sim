# Set up logging functionality
import logging

logger_name = "DSRC"

logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.debug("Set up logger for DSRC")
