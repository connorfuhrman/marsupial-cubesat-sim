"""Create an animation of a simulation."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import groupby
import numpy as np
from DSRC.simulation import SimulationHistory
from tqdm import tqdm
from itertools import chain


class _animation_tqdm(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, curr_frame, total_frame):
        return self.update(total_frame - curr_frame)


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _update_plot(frame_num, datagetter, axs):
    all_lines = []
    for ax in axs:
        lines, sample_lines, frame_data, metadata = datagetter(ax, frame_num)
        if frame_data is not None and metadata is not None:
            positions = frame_data['craft_positions']
            for id, line in lines.items():
                if id in positions:
                    marker = "$c$" if frame_data['craft_types'][id] == "CubeSat" else "$M$"
                    line.set_data_3d(*positions[id])
                    line.set(alpha=1.0)
                    line.set(marker=marker)
                else:
                    line.set(alpha=0.0)

            for i, pos in enumerate(frame_data['sample_positions']):
                sample_lines[i].set_data_3d(*pos)
                sample_lines[i].set(alpha=1.0, color='b')
            i = len(frame_data['sample_positions'])
            while i < len(sample_lines):
                sample_lines[i].set(alpha=0.0)
                i += 1
            ax.set_title(f"Simulation {metadata['id']}\nat iteration {frame_num} "
                         f" (time: {float(frame_data['time']):.5}s)")
        else:
            for line in chain(lines.values(), sample_lines):
                line.set(alpha=0.0)
            ax.set_title("Simulation concluded")
        all_lines.extend(lines.values())
        all_lines.extend(sample_lines)

    return all_lines


def _get_plot_layout(nplots: int) -> tuple[int, int]:
    if nplots == 1:
        return 1, 1
    elif (sqrt := np.sqrt(nplots)) == round(sqrt):
        return int(sqrt), int(sqrt)
    else:
        return int(f := np.floor(sqrt)), int(nplots - f)


def entrypoint(sim_history: list[SimulationHistory], fname: str = None):
    """Animate the results of a simulation."""
    fig = plt.figure(figsize=(25, 15))
    ax_map = dict()
    max_max_iters = 0
    nplot_rows, nplot_cols = _get_plot_layout(len(sim_history))
    for i, h in enumerate(sim_history):
        ax = fig.add_subplot(nplot_rows, nplot_cols, i + 1, projection="3d")

        def do_plot(marker):
            return ax.plot([], [], [], alpha=0.0, marker=marker)[0]

        ax.set(xlim3d=(-25, 25),
               ylim3d=(-25, 25),
               zlim3d=(-25, 25))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax_map[ax] = {
            "sim_data": h,
            "lines": {id: do_plot(marker="s")
                      for id in h['metadata']['craft_ids']},
            "sample_lines": [do_plot(marker=".")
                             for _ in range(h['metadata']['max_num_samples'])]
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
        return data['lines'], data['sample_lines'], history, metadata

    ani = FuncAnimation(fig,                        # noqa F481: I know this is unused
                        _update_plot,
                        max_max_iters+1,
                        fargs=(getdata, list(ax_map.keys()),),
                        interval=10)
    if fname is not None:
        with _animation_tqdm(total=max_max_iters, desc="Animation MP4 Save") as t:
            ani.save(fname, progress_callback=t.update_to)
    else:
        plt.show()
