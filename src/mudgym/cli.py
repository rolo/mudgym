"""Interactive command-line client for a single MUD session."""

import argparse
import sys

from mudgym.connections.connection import MudConnection
from mudgym.connections.registry import available_connections_dict, default_connection
from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.featurizers.strings import decode_text_bytes
from mudgym.logs import setup_logging
from mudgym.session import MudSession


def play(connection_class: type[MudConnection] = default_connection) -> None:
    print("Beginning game session with connection: ", connection_class.__name__)
    session = MudSession(
        connection=connection_class(),
        observation_line=FEInventoryField.command,
        end_of_turn_marker=FEInventoryField.end_of_turn_marker,
    )
    session.reset()

    raw_bytes, terminated, incomplete, _ = session.command("l")
    if raw_bytes:
        print(decode_text_bytes(raw_bytes), end="")
    print()
    sys.stdout.flush()

    while not (terminated or incomplete):
        raw_bytes, terminated, incomplete, _ = session.command(input())
        if raw_bytes:
            print(decode_text_bytes(raw_bytes), end="")
        print()
        sys.stdout.flush()

    print("Exiting...")
    session.close()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Play a MUD session")
    parser.add_argument(
        "--connection",
        choices=list(available_connections_dict),
        metavar="SLUG",
        help=f"Connection type to use. Available: {', '.join(available_connections_dict)}",
    )
    args = parser.parse_args()
    play(available_connections_dict[args.connection] if args.connection else default_connection)


if __name__ == "__main__":
    main()
