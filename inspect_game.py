from pathlib import Path

import chess
import chess.pgn


USERNAME = "ollervo"

PGN_PATH = (
    Path(__file__).resolve().parent
    / "chess_data"
    / USERNAME
    / "chesscom"
    / f"{USERNAME}_chesscom.pgn"
)

# Read the first game from the file.
with PGN_PATH.open(encoding="utf-8-sig") as pgn_file:
    game = chess.pgn.read_game(pgn_file)

if game is None:
    raise ValueError("The PGN file contains no games.")

if game.errors:
    raise ValueError(f"The game contains parsing errors: {game.errors}")

# Determine which colour you played.
if game.headers.get("White", "").casefold() == USERNAME.casefold():
    player_colour = chess.WHITE
elif game.headers.get("Black", "").casefold() == USERNAME.casefold():
    player_colour = chess.BLACK
else:
    raise ValueError("You are not listed as a player in this game.")

board = game.board()

if type(board) is not chess.Board or board.chess960:
    raise ValueError("This example expects standard chess.")

print("White:", game.headers.get("White"))
print("Black:", game.headers.get("Black"))
print("Date:", game.headers.get("Date"))
print("Your colour:", "White" if player_colour else "Black")

examples = []

# Replay both players' moves, recording only your decisions.
for move in game.mainline_moves():
    if board.turn == player_colour:
        example = {
            "fen": board.fen(),
            "move_uci": move.uci(),
            "move_san": board.san(move),
            "legal_moves": [
                candidate.uci()
                for candidate in board.legal_moves
            ],
        }

        examples.append(example)

    # Advance the board AFTER recording the position.
    board.push(move)

print(f"\nExtracted {len(examples)} of your decisions.")

for number, example in enumerate(examples[:5], start=1):
    print(f"\nExample {number}")
    print("Position:", example["fen"])
    print("Your move:", example["move_san"])
    print("Move in UCI:", example["move_uci"])
    print("Legal alternatives:", example["legal_moves"])