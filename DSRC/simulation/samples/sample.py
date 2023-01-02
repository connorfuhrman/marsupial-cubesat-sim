"""A sample to be collected."""

import numpy as np


class Sample:
    """Standin as a sample class for now."""

    _weight: float = 10.0
    """Sample weight in g."""
    _value: float = 5.0
    """Value in [0, 10] rating how valuable the sample is."""
    _position: np.ndarray = None
    """The sample's location in 3-space."""
    _velocity: np.ndarray = np.zeros(3)
    """Samples velocity in 3-space."""

    def __init__(self, weight, value, position, velocity):  # noqa D
        self._weight = weight
        self._value = value
        self._position = position
        self._velocity = velocity

    def update_kinematics(self, dt: float, *args) -> None:
        """Update the position of the sample."""
        self._position += float(dt) * self._velocity
        return True  # Return value adhere to entity specification for simulation

    def __eq__(self, o):  # noqa D
        return (
            self.weight == o.weight
            and self.value == o.value
            and np.array_equal(self.position, o.position)
            and np.array_equal(self.velocity, o.velocity)
        )

    def __key(self):
        """Make a hashing key.

        The tuple type is hashable so long as it contains
        primitive data types (I think) but here a unique
        hash is created to allow this type to be used in
        sets and dicts.
        """
        return (self.weight, self.value, *self.position, *self.velocity)

    def __hash__(self):  # noqa D
        return hash(self.__key())

    @property
    def weight(self) -> float:  # noqa D
        return self._weight

    @property
    def value(self) -> float:  # noqa D
        return self._value

    @property
    def position(self) -> np.ndarray:  # noqa D
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:  # noqa D
        return self._velocity.copy()
