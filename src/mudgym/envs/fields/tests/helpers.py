import numpy as np
from gymnasium import spaces


def assert_valid_observation(field, obs):
    declared = field.space()
    assert obs.keys() == declared.keys()

    for key, space in declared.items():
        value = obs[key]

        if isinstance(space, spaces.Box | spaces.Discrete | spaces.MultiBinary | spaces.MultiDiscrete):
            assert isinstance(value, np.ndarray | np.generic), (
                f"{key}: expected a NumPy value, got {type(value).__name__}"
            )
            assert value.dtype == space.dtype, f"{key}: emitted {value.dtype}, declared {space.dtype}"

        assert space.contains(value), f"{key}: {value!r} not in {space}"
