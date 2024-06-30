import torch as t
from heuristics_base_v2 import (
    EMPTY,
    PAWN,
    UNMOVED_ROOK,
    KNIGHT,
    LIGHT_BISHOP,
    DARK_BISHOP,
    QUEEN,
    UNMOVED_KING,
    OPPONENT_OFFSET,
)

# fmt: off
DEFAULT_STARTING_BOARD = t.tensor([
    [UNMOVED_ROOK, KNIGHT, LIGHT_BISHOP, QUEEN, UNMOVED_KING, DARK_BISHOP, KNIGHT, UNMOVED_ROOK],
    [PAWN, PAWN, PAWN, PAWN, PAWN, PAWN, PAWN, PAWN],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [PAWN+OPPONENT_OFFSET, PAWN+OPPONENT_OFFSET, PAWN+OPPONENT_OFFSET, PAWN+OPPONENT_OFFSET, PAWN+OPPONENT_OFFSET, PAWN+OPPONENT_OFFSET, PAWN+OPPONENT_OFFSET, PAWN+OPPONENT_OFFSET],
    [UNMOVED_ROOK+OPPONENT_OFFSET, KNIGHT+OPPONENT_OFFSET, LIGHT_BISHOP+OPPONENT_OFFSET, QUEEN+OPPONENT_OFFSET, UNMOVED_KING+OPPONENT_OFFSET, DARK_BISHOP+OPPONENT_OFFSET, KNIGHT+OPPONENT_OFFSET, UNMOVED_ROOK+OPPONENT_OFFSET]
])
# fmt: on