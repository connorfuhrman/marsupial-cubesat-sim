"""Simulation of Decentralized Sample Recovery Coordination."""


from DSRC import logger_name
from DSRC.simulation import (
    animate_simulation,
    logger_name as simulation_logger_name
)
from DSRC.simulation.spacecraft import CubeSat
from functools import partial
from dataclasses import dataclass
import numpy as np
import logging
from copy import copy

logger: logging.Logger = logging.getLogger(f"{logger_name}.{simulation_logger_name}")


def _entity_dict_entry() -> dict:
    """Get a blank initial entity dict entry."""
    return {
        "position": [],
        "velocity": [],
        "orientation": [],
        "time": []
    }


def _update_entity_dict(entities: dict, entity, time):
    e = entities[entity.id]
    e["time"].append(time)
    e["position"].append(entity.position)
    e["velocity"].append(entity.velocity)
    e["orientation"].append(entity.orientation)


@dataclass
class Sample:
    """Standin as a sample class for now."""

    weight: float = 10.0
    """Sample weight in g."""
    value: float = 5.0
    """Value in [0, 10] rating how valuable the sample is."""
    position: np.ndarray = None
    """The sample's location in 3-space."""


def distance_btwn(a, b):
    """Get the magnitude of the distance between two things."""
    return np.linalg.norm(a.position - b.position)


def make_circle(center: np.ndarray, rad: float, npoints: int = 100) -> np.ndarray:
    """Make a circle centered around some point with a given radius.

    The circle lies in a horizontal plane at z = center[2].
    """
    theta = np.linspace(0, 2*np.pi, npoints)
    xpnts = rad * np.cos(theta)
    ypnts = rad * np.sin(theta)

    arr = np.empty((npoints, 3), dtype=float)
    for i, (x, y) in enumerate(zip(xpnts, ypnts)):
        arr[i, :] = np.array([x, y, center[2]])
    return arr


def run_simulation(max_time: float, dt: float) -> dict:
    """Run a simulation for some number of seconds.

    max_time is the maximum sim time in seconds
    dt is the update period in seconds.

    Returns dictionary mapping craft IDs to their history.
    """
    entities = {}

    # Start at origin with velocity in x dir of 10 cm per second
    craft = CubeSat(loc=[0, 0, 0],
                    fuel_level=100,
                    sample_prob=0.95)
    entities[craft.id] = _entity_dict_entry()
    entities["particle"] = {
        "time": [],
        "position": []
    }

    particle = Sample(position=[-3, 0, 0])

    # Calculate a circle trajectory and set as waypoints
    # to follow
    waypoints = make_circle(np.array([0, 0, 0]), 6, 15)
    for w in waypoints:
        craft.add_waypoint(w)

    current_time = 0.0

    update = partial(_update_entity_dict, entities, craft)
    update(current_time)
    sample_collection_attempted = False
    while current_time <= max_time:
        craft.update_kinematics(dt)
        if distance_btwn(craft, particle) < 0.5 and \
           not craft.has_sample and \
           not sample_collection_attempted:
            logger.info("Attempting to capture sample")
            craft.attempt_sample_capture(particle)
            sample_collection_attempted = True
            if craft.has_sample:
                logger.info("Sample captured!!")
            else:
                logger.info("Sample was missed!!")
        current_time += dt
        update(current_time)
        if not craft.has_sample:
            entities["particle"]["time"].append(current_time)
            entities["particle"]["position"].append(particle.position)

    return entities


if __name__ == '__main__':
    try:
        entities_dict = run_simulation(100, 0.25)
        print("Animating ...")
        animate_simulation(entities_dict)
    except:  # noqa E772
        import pdb, traceback, sys  # noqa E401
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
