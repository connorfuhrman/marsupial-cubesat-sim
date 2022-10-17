"""Create an animation of a simulation."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import groupby


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _update_plot(frame_num: int, entities: dict, lines: list):
    for line, entity in zip(lines, entities.values()):
        pos = entity["position"][frame_num]
        line.set_data_3d(*pos)
    return lines


def entrypoint(entities: dict):
    """Animate the results of a simulation.

    Pass the dictionary returned by the simulation
    to this function (imported by default as DSRC.animate_simulation).
    """
    if len(entities) == 0:
        raise ValueError("Got empty dictionary of entities!")
    # Get the total number of simulation timesteps
    num_timesteps = [len(e["time"]) for e in entities.values()]
    if not _all_equal(num_timesteps):
        raise ValueError("The number of timesteps is inconsistent "
                         f"in your dictionary: {num_timesteps}")
    num_timesteps = num_timesteps[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim3d=(-10, 10),
           ylim3d=(-10, 10),
           zlim3d=(-10, 10))

    lines = [ax.plot([], [], [],  marker="o")[0] for _ in entities]

    ani = FuncAnimation(fig,                        # noqa F481: I know this is unused
                        _update_plot,
                        num_timesteps,
                        fargs=(entities, lines,))

    plt.show()
