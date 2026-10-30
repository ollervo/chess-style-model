import json
from pathlib import Path

import chess
import chess.pgn


USERNAME = "ollervo"

DATA_DIR = (
    Path(__file__).resolve().parent
    / "chess_data"
    / USERNAME
)

PGN_FILES = {
    "chesscom": DATA_DIR / "chesscom" / f"{USERNAME}_chesscom.pgn",
    "lichess": DATA_DIR / "lichess" / f"{USERNAME}_lichess.pgn",
}

OUTPUT_PATH = DATA_DIR / "decisions.jsonl"


def extract_examples(game, source, game_number):
    """Return this game's decisions, or None if we should skip it."""

    if game.errors:
        return None

    # Identify your colour.
    if game.headers.get("White", "").casefold() == USERNAME.casefold():
        player_colour = chess.WHITE
    elif game.headers.get("Black", "").casefold() == USERNAME.casefold():
        player_colour = chess.BLACK
    else:
        return None

    board = game.board()

    # Exclude variants for now.
    if type(board) is not chess.Board or board.chess960:
        return None

    player_rating_key = (
        "WhiteElo" if player_colour == chess.WHITE else "BlackElo"
    )

    # This ID groups all decisions from the same game together.
    game_id = f"{source}:{game_number}"

    examples = []

    for ply, move in enumerate(game.mainline_moves()):
        if board.turn == player_colour:
            examples.append({
                "game_id": game_id,
                "source": source,
                "date": game.headers.get("Date"),
                "time_control": game.headers.get("TimeControl"),
                "player_rating": game.headers.get(player_rating_key),
                "player_colour": (
                    "white" if player_colour == chess.WHITE else "black"
                ),
                "ply": ply,
                "fen": board.fen(),
                "move_uci": move.uci(),
                "legal_moves": [
                    candidate.uci()
                    for candidate in board.legal_moves
                ],
            })

        board.push(move)

    return examples


def main():
    # Check both files before creating the output.
    for path in PGN_FILES.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing PGN file: {path}")

    total_examples = 0
    temporary_path = OUTPUT_PATH.with_suffix(".jsonl.part")

    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for source, pgn_path in PGN_FILES.items():
                games_read = 0
                games_used = 0
                games_skipped = 0
                source_examples = 0

                with pgn_path.open(encoding="utf-8-sig") as pgn_file:
                    while True:
                        game = chess.pgn.read_game(pgn_file)

                        if game is None:
                            break

                        games_read += 1

                        examples = extract_examples(
                            game,
                            source,
                            games_read,
                        )

                        if not examples:
                            games_skipped += 1
                            continue

                        for example in examples:
                            output.write(
                                json.dumps(example) + "\n"
                            )

                        games_used += 1
                        source_examples += len(examples)

                        if games_read % 1000 == 0:
                            print(
                                f"{source}: {games_read:,} games read, "
                                f"{source_examples:,} decisions",
                                flush=True,
                            )

                total_examples += source_examples

                print(
                    f"{source} finished: "
                    f"{games_used:,} games used, "
                    f"{games_skipped:,} skipped, "
                    f"{source_examples:,} decisions"
                )

        # Replace the dataset only after the complete run succeeds.
        temporary_path.replace(OUTPUT_PATH)

    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"\nTotal decisions: {total_examples:,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()