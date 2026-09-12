"""Play a WASM episode through the installed MudGym wheel's default connection."""

import mudgym


def main() -> None:
    print("mudgym", mudgym.__version__)
    with mudgym.make_env(observation="parsed") as env:
        observation, info = env.reset(seed=123)
        assert env.observation_space.contains(observation)
        observation, reward, terminated, truncated, info = env.step("look")
        assert not terminated and not truncated
        assert env.observation_space.contains(observation)
        print("wasm", observation["text"][:120])
    print("smoke OK")


if __name__ == "__main__":
    main()
