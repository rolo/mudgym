"""
MudGym: a reinforcement learning environment for MUD2.

Importing this package registers the Gymnasium env IDs.
"""

from importlib.metadata import PackageNotFoundError, version

from mudgym.envs.factory import make_env, make_parallel_env, make_vector_env
from mudgym.envs.registration import register_envs as _register_envs

try:
    __version__ = version("mudgym")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"

_register_envs()

__all__ = [
    "__version__",
    "make_env",
    "make_parallel_env",
    "make_vector_env",
]
