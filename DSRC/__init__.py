# Set up logging functionality
import logging

logger = logging.getLogger("DSRC")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.debug("Set up logger for DSRC")

from .simulation.spacecraft import Spacecraft
from .simulation.animation import entrypoint as animate_simulation
