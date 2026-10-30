import chess
import numpy as np


def encode_position(fen):
    """Convert a standard chess position into a (19, 8, 8) array."""

    board = chess.Board(fen)

    planes = np.zeros((19, 8, 8), dtype=np.float32)

    # Piece layers.
    for square, piece in board.piece_map().items():
        colour_offset = 0 if piece.color == chess.WHITE else 6

        # python-chess numbers piece types from pawn=1 to king=6.
        layer = colour_offset + piece.piece_type - 1

        # Array row 0 represents rank 8; column 0 represents file a.
        row = 7 - chess.square_rank(square)
        column = chess.square_file(square)

        planes[layer, row, column] = 1.0

    # Side to move.
    planes[12].fill(float(board.turn == chess.WHITE))

    # Castling rights.
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]

    for layer, has_right in enumerate(castling_rights, start=13):
        planes[layer].fill(float(has_right))

    # En passant target, if present.
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        column = chess.square_file(board.ep_square)

        planes[17, row, column] = 1.0

    # Moves since the last pawn move or capture, measured in plies.
    planes[18].fill(board.halfmove_clock / 100.0)

    return planes

if __name__ == "__main__":
    import json
    from pathlib import Path

    dataset_path = (
        Path(__file__).resolve().parent
        / "chess_data"
        / "ollervo"
        / "decisions.jsonl"
    )

    with dataset_path.open(encoding="utf-8") as file:
        example = json.loads(next(file))

    encoded = encode_position(example["fen"])

    print("Position:")
    print(chess.Board(example["fen"]))

    print("\nArray shape:", encoded.shape)