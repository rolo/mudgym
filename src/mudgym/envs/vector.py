from gymnasium.vector import SyncVectorEnv, VectorWrapper

from mudgym.connections.provider import ConnectionProvider


class MudVectorEnv(VectorWrapper):
    """
    Manages the ConnectionProvider lifecycle and provides index access to the underlying environments

    Note: This wrapper assumes a SyncVectorEnv underneath. The `__getitem__` method accesses `.env.envs[index]` which is specific to SyncVectorEnv.
    """

    def __init__(
        self,
        env: SyncVectorEnv,
        provider: ConnectionProvider,
    ):
        super().__init__(env)
        self._provider = provider

    def close(self, **kwargs):
        try:
            super().close(**kwargs)
        finally:
            self._provider.close()

    def __getitem__(self, index: int):
        """Access individual sub-environments by index.

        Raises:
            IndexError: If index is out of range.
            AttributeError: If underlying env doesn't have .envs (not SyncVectorEnv).
        """
        cur = self.env
        while hasattr(cur, "env") and not hasattr(cur, "envs"):
            cur = cur.env
        envs = getattr(cur, "envs", None)
        if envs is None:
            raise AttributeError(
                f"Underlying {type(self.env).__name__} has no .envs attribute. "
                "MudVectorEnv.__getitem__ only works with SyncVectorEnv."
            )
        return envs[index]
