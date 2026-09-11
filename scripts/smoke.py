"""Play Docker and WASM episodes against the installed MudGym wheel."""

import mudgym


def main() -> None:
    print("mudgym", mudgym.__version__)
    for connection in ("docker_run", "wasm"):
        with mudgym.make_env(connection=connection, observation="parsed") as env:
            observation, info = env.reset(seed=123)
            assert env.observation_space.contains(observation)
            observation, reward, terminated, truncated, info = env.step("look")
            assert not terminated and not truncated
            assert env.observation_space.contains(observation)
            print(connection, observation["text"][:120])
    print("smoke OK")


if __name__ == "__main__":
    main()
