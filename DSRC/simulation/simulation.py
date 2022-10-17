"""Simulation of Decentralized Sample Recovery Coordination."""


from DSRC import Spacecraft, animate_simulation
from functools import partial


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


def run_simulation(max_time: float, dt: float) -> dict:
    """Run a simulation for some number of seconds.

    max_time is the maximum sim time in seconds
    dt is the update period in seconds.

    Returns dictionary mapping craft IDs to their history.
    """
    entities = {}

    # Start at origin with velocity in x dir of 10 cm per second
    craft = Spacecraft([0, 0, 0], [0.5, 0, 0])
    entities[craft.id] = _entity_dict_entry()

    current_time = 0.0

    update = partial(_update_entity_dict, entities, craft)
    update(current_time)
    while current_time <= max_time:
        craft.update_kinematics(dt)
        current_time += dt
        update(current_time)

    return entities


if __name__ == '__main__':
    entities_dict = run_simulation(5, 0.5)
    animate_simulation(entities_dict)
