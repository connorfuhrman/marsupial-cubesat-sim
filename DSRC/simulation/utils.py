"""Various utilities for interacting with a simulation.

This file is a WIP. It might get deleted.
"""

import numpy as np
import json
from mpmath import mpf


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, mpf):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def to_json(obj):
    """Wrapper around json.dumps with a numpy-aware encoder.

    Objects of type np.ndarray will be converted to a list.
    """
    return json.dumps(obj, cls=_Encoder)


def save_json_file(obj, filename):
    """Save a JSON to a file with a numpy-aware encoder."""
    with open(filename, "w") as f:
        json.dump(obj, f, cls=_Encoder)


def distance_to_samples(craft, samples) -> list:
    """Get the distance to all samples."""
    return [np.linang.norm(craft.position, s.position) for s in samples]
