from contextlib import closing

import pytest

from mudgym import make_env, make_parallel_env
from mudgym.connections.persona import Persona
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.featurizers.ansi import strip_ansi


@pytest.mark.parametrize("sex", ["male", "female"])
def test_engine_generates_stats_for_the_selected_sex(wasm_runtime, sex):
    stats = []
    for seed in (123, 456):
        world = wasm_runtime.create_world(max_players=1, seed=seed)
        try:
            player = world.add_session(Persona("Ada", sex))
            output = strip_ansi(player.send("score", 5000).output).decode("latin-1")
            fields = dict(line.split(":", 1) for line in output.splitlines() if ":" in line)
            assert fields["sex"].strip() == player.persona.sex == sex
            stats.append(tuple(int(fields[field].split()[0]) for field in ("strength", "dexterity", "stamina")))
            assert all(value > 0 for value in stats[-1])
        finally:
            world.shutdown()
    assert stats[0] != stats[1]


@pytest.mark.parametrize("persona", [Persona(), Persona("Ada"), Persona(sex="f")])
def test_admission_requires_a_resolved_persona(wasm_runtime, persona):
    world = wasm_runtime.create_world(max_players=1, seed=123)
    try:
        with pytest.raises(ValueError, match="resolved persona"):
            world.add_session(persona)
        assert world._call("mud2_shared_player_count") == 0
    finally:
        world.shutdown()


@pytest.mark.parametrize(
    ("options", "name", "sex"),
    [
        ({}, None, None),
        ({"persona": "Dave"}, "Dave", None),
        ({"sex": "f"}, None, "female"),
        ({"persona": "Davina", "sex": "F"}, "Davina", "female"),
        ({"sex": "m", "persona_pool": [("Davina", "f"), ("Dave", "male")]}, "Dave", "male"),
        ({"sex": "female", "persona_pool": [("Ash", None)]}, "Ash", "female"),
    ],
)
def test_scalar_factory_passes_persona_options(wasm_runtime, options, name, sex):
    with closing(make_env(**options, connection_kwargs={"runtime": wasm_runtime})) as env:
        _, info = env.reset(seed=123)
        identity = info["transport"]["persona"]
        assert identity == {"name": info["persona"], "sex": info["persona_sex"]}
        if name is not None:
            assert info["persona"] == name
        if sex is not None:
            assert info["persona_sex"] == sex
        stepped = env.step("look")[-1]
        assert stepped["persona"] == info["persona"]
        assert stepped["persona_sex"] == info["persona_sex"]
        assert stepped["transport"]["persona"] == identity


@pytest.mark.parametrize("options", [{}, {"persona": "Davina", "sex": "f"}])
def test_scalar_identity_repeats_after_reseeding_and_reset_without_a_seed(wasm_runtime, options):
    with closing(make_env(**options, connection_kwargs={"runtime": wasm_runtime})) as env:
        runs = []
        for seed in (123, 456, 123, None):
            obs, info = env.reset(seed=seed)
            assert info["transport"]["world_seed"] == (123 if seed is None else seed)
            if options:
                assert (info["persona"], info["persona_sex"]) == ("Davina", "female")
            runs.append((obs["text"], info["persona"], info["persona_sex"]))
        assert runs[0] == runs[2] == runs[3]


def test_scalar_persona_options_override_connection_kwargs(wasm_runtime):
    with closing(
        make_env(
            sex="f",
            persona_pool=[("Davina", "f")],
            connection_kwargs={"runtime": wasm_runtime, "sex": "m", "persona_pool": [("Dave", "m")]},
        )
    ) as env:
        info = env.reset(seed=42)[1]
        assert (info["persona"], info["persona_sex"]) == ("Davina", "female")


@pytest.mark.parametrize(
    ("personas", "pool", "expected"),
    [
        ([("Davina", "f"), ("Dave", "male")], None, [("Davina", "female"), ("Dave", "male")]),
        ([("Aaron", "m"), (None, "f")], [("Freya", "f")], [("Aaron", "male"), ("Freya", "female")]),
    ],
)
def test_parallel_factory_passes_personas_and_pool(personas, pool, expected):
    with closing(make_parallel_env(2, personas=personas, persona_pool=pool)) as env:
        _, infos = env.reset(seed=42)
        assert [(infos[agent]["persona"], infos[agent]["persona_sex"]) for agent in env.possible_agents] == expected


def test_parallel_factory_accepts_name_and_sex_shorthand():
    with closing(
        make_parallel_env(
            3,
            personas=[("Dave",), ("F",), ("male",)],
            persona_pool=[("Freya", "f"), ("Aaron", "m")],
        )
    ) as env:
        _, infos = env.reset(seed=42)
        assert infos["player_0"]["persona"] == "Dave"
        assert infos["player_0"]["persona_sex"] in {"male", "female"}
        assert (infos["player_1"]["persona"], infos["player_1"]["persona_sex"]) == ("Freya", "female")
        assert (infos["player_2"]["persona"], infos["player_2"]["persona_sex"]) == ("Aaron", "male")


def test_replacement_keeps_female_identity_in_existing_world(wasm_runtime):
    with closing(
        WasmtimeProvider(runtime=wasm_runtime, worlds=1, personas=[("Freya", "f"), ("Aaron", "m")])
    ) as provider:
        connection, peer = provider.create_connections(2)
        provider.reset(seed=123)
        connection.reset()
        peer.reset()
        previous, peer_session = connection._session, peer._session
        connection.send_line("quit")
        assert connection.read_response()[1]
        connection.reset()
        assert connection._session is not previous
        assert connection._session.world is previous.world
        assert peer._session is peer_session
        assert connection._session.persona == previous.persona == Persona("Freya", "female")
        connection.send_line("score")
        assert b"sex:            female" in strip_ansi(connection.read_response()[0])


def test_persona_count_must_match_connection_count(wasm_runtime):
    with closing(WasmtimeProvider(runtime=wasm_runtime, personas=[("Dave",)])) as provider:
        with pytest.raises(ValueError, match="personas"):
            provider.create_connections(2)
        assert provider._ordered_worlds == []


def test_insufficient_pool_fails_before_admission(wasm_runtime):
    with closing(WasmtimeProvider(runtime=wasm_runtime, worlds=1, persona_pool=[("Freya", "f")])) as provider:
        provider.create_connections(2)
        with pytest.raises(ValueError, match="pool"):
            provider.reset(seed=42)
        assert provider._ordered_worlds == []


def test_interleaved_worlds_keep_overrides_local_and_reseed_independently(wasm_runtime):
    requested = [("Ash", "m"), ("Ash", "f"), ("f",), ("m",)]
    with closing(WasmtimeProvider(runtime=wasm_runtime, worlds=2, personas=requested)) as provider:
        connections = provider.create_connections(4)
        provider.reset(seed=42)
        for connection in connections:
            connection.reset()
        identities = [connection._session.persona for connection in connections]
        assert identities[:2] == [Persona("Ash", "male"), Persona("Ash", "female")]
        assert identities[2].sex == "female" and identities[3].sex == "male"
        assert identities[2].name != "Ash" and identities[3].name != "Ash"
        provider.reset(seed=[None, 45, None, None])
        assert identities[::2] == [connection._session.persona for connection in connections[::2]]
