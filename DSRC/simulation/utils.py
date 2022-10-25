"""Various utilities for interacting with a simulation."""

import numpy as np


def distance_to_samples(craft, samples) -> list:
    """Get the distance to all samples."""
    return [np.linang.norm(craft.position, s.position) for s in samples]
