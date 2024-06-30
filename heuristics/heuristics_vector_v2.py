from __future__ import annotations
import jaxtyping
import torch as t
from typing import List

from heuristics_base_v2 import (
    BaseEvaluator,
    EMPTY,
    PAWN,
    PASSANT_PAWN,
    UNMOVED_ROOK,
    MOVED_ROOK,
    KNIGHT,
    LIGHT_BISHOP,
    DARK_BISHOP,
    QUEEN,
    UNMOVED_KING,
    MOVED_KING,
    OPPONENT_OFFSET,
)


class BoardVectorEvaluator(BaseEvaluator):
    """
    A vector evaulator is also a valuator but takes in a board and outputs a
    single vector of values that represent some compressed representation valuation.
    Normally these will be singleton values, so this can be thought of as a single
    "estimate" for the board.

    NOTE that if you want to do evaluation by quadrant or something like this, this
    is your answer. Just squeeze the lower res. board.
    """

    def evaulate(
        self, board: jaxtyping.Float[t.Tensor, "batch 8 8 1"]
    ) -> jaxtyping.Float[t.Tensor, "batch depth"]:
        raise NotImplementedError("Please implement this board validator.")


class ConcatBoardVectorEvaluator(BoardVectorEvaluator):
    # NOTE this is a copy of `ConcatBoardTabularEvaluator` but we include it
    # here just so we can get jaxtyping
    def __init__(self, *valuators: List[BoardVectorEvaluator]) -> None:
        self.valuators = valuators

    def evaluate(
        self, board: jaxtyping.Float[t.Tensor, "batch 8 8 1"]
    ) -> jaxtyping.Float[t.Tensor, "batch depth"]:
        return t.cat([val(board) for val in self.valuators], dim=-1)


################ EVALUATORS ################
class InverseMeanDistanceFromKing(BoardVectorEvaluator):
    def __init__(
        self,
        my: bool = False,
        opponent: bool = False,
        only_include_pieces: bool = False,
        default_value_for_empty: float = 0.0,
    ) -> None:
        super().__init__(my, opponent, only_include_pieces, default_value_for_empty)

    def evaluate(
        self, board: jaxtyping.Float[t.Tensor, "batch 8 8 1"]
    ) -> jaxtyping.Float[t.Tensor, "batch 1"]:
        raise NotImplementedError("Please implement this board validator.")

class PawnCount(BoardVectorEvaluator):
    pass # XXX

class PieceCount(BoardVectorEvaluator):
    pass  # XXX calculate tabular first
# XXX do this above with a flag
# class PieceValuesSum(BoardVectorEvaluator):
#     """Sum of standard values: 1, 3, 3, 5, 9."""

#     pass  # XXX NOTE that by having the masks of what pieces are were, we should be getting some decent information


class OpenFilesToKing(BoardVectorEvaluator):
    pass  # XXX calculate tabular first


class OpenRanksToKing(BoardVectorEvaluator):
    pass  # XXX calculate tabular first


class OpenDiagonalsToKing(BoardVectorEvaluator):
    pass  # XXX calculate tabular first


class NumDoubledBlockedPawns(BoardVectorEvaluator):
    pass  # XXX calculate tabular first


class NumPassedPawns(BoardVectorEvaluator):
    pass  # XXX calculate tabular first


class NumIsolatedPawns(BoardVectorEvaluator):
    def __init__(
        self,
        my: bool = False,
        opponent: bool = False,
        only_include_pieces: bool = False,
        default_value_for_empty: float = 0,
        count_with_no_on_adj_files: bool = True,
        count_with_no_on_surrounding_square: bool = False,
    ) -> None:
        super().__init__(my, opponent, only_include_pieces, default_value_for_empty)
        self.count_with_no_on_adj_files = count_with_no_on_adj_files
        self.count_with_no_on_surrounding_square = count_with_no_on_surrounding_square

    pass  # XXX calculate tabular first

class CentralControl(BoardVectorEvaluator):
    """Return a vector that has the values for the central squares stacked, where it says who has which."""
    pass  # XXX calculate tabular first

class NumBlackOrWhiteDefended(BoardVectorEvaluator):
    """Helps you detect if you might have week defense to a bishop."""
    pass  # XXX calculate tabular first

class AlivePieces(BoardVectorEvaluator):
    """Return a vector that says if each piece is alive or not."""
    pass  # XXX calculate tabular first

class PawnWallLength(BoardVectorEvaluator):
    """Return a vector that says how long the pawn wall is."""
    pass  # XXX calculate tabular first

class KingsCastled(BoardVectorEvaluator):
    """Return a vector that says if each king is castled and on which side."""
    pass  # XXX 

class CanCastle(BoardVectorEvaluator):
    """
    Return a vector that says if each king can castle and whether castle is blocked
    by an attack (likely indicative of bad value).
    """
    pass  # XXX

class NumKingDefenders(BoardVectorEvaluator):
    pass  # XXX

class NumPiecesForwards(BoardVectorEvaluator):
    """
    How many pieces are on the enemy side of the board.
    """
    pass  # XXX

# XXX coordination?
# XXX 