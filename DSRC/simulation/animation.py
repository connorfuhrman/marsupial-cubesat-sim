"""Create an animation of a simulation."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import groupby
import numpy as np
from DSRC.simulation import SimulationHistory


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _update_plot(frame_num, datagetter, axs):
    for ax in axs:
        lines, frame_data, metadata = datagetter(ax, frame_num)
        if frame_data is not None and metadata is not None:
            positions = frame_data['craft_positions']
            for id, line in lines.items():
                if id in positions:
                    line.set_data_3d(*positions[id])
                    line.set(alpha=1.0)
                else:
                    line.set(alpha=0.0)
            ax.set_title(f"Simulation {metadata['id']}\nat iteration {frame_num}")
        else:
            for line in lines:
                line.set(alpha=0.0)
            ax.set_title("Simulation concluded")

    return lines


def _get_plot_layout(nplots: int) -> tuple[int, int]:
    if nplots == 1:
        return 1, 1
    elif (sqrt := np.sqrt(nplots)) % 2.0 == 0:
        return int(sqrt), int(sqrt)
    else:
        return int(f := np.floor(sqrt)), int(nplots - f)


def entrypoint(sim_history: list[SimulationHistory]):
    """Animate the results of a simulation."""
    fig = plt.figure()
    ax_map = dict()
    max_max_iters = 0
    nplot_rows, nplot_cols = _get_plot_layout(len(sim_history))
    for i, h in enumerate(sim_history):
        ax = fig.add_subplot(nplot_rows, nplot_cols, i + 1, projection="3d")
        ax.set(xlim3d=(-10, 10),
               ylim3d=(-10, 10),
               zlim3d=(-20, 5))
        lines = {id: ax.plot([], [], [],  marker="o", alpha=0.0)[0]
                 for id in h['metadata']['craft_ids']}
        ax_map[ax] = {
            "sim_data": h,
            "lines": lines
        }
        if (ti := h['metadata']['total_iters']) > max_max_iters:
            max_max_iters = ti

    def getdata(ax, framenum):
        data = ax_map[ax]
        try:
            history = data['sim_data']['history'][framenum]
            metadata = data['sim_data']['metadata']
        except IndexError:
            history = None
            metadata = None
        return data['lines'], history, metadata

    ani = FuncAnimation(fig,                        # noqa F481: I know this is unused
                        _update_plot,
                        max_max_iters,
                        fargs=(getdata, list(ax_map.keys()),),
                        interval=10)

    plt.show()
