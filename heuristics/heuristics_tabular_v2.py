from jaxtyping._indirection import Float as Float
import torch as t
import torch.functional as F
import jaxtyping
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

################ Valuator ################


class BoardTabularEvaluator(BaseEvaluator):
    """
    A board evaluator is a class that really is meant to wrap a single function: evaluate(...)
    which returns a per-cell value for the value of a board. It is meant, specifically, for
    SPACIAL valuations, based on where pieces are, etc...

    The right approach is to subclass this to create a bunch of valuators.
    """

    def evaluate(
        self, board: jaxtyping.Float[t.Tensor, "batch 8 8 1"]
    ) -> jaxtyping.Float[t.Tensor, "batch 8 8 depth"]:
        raise NotImplementedError("Please implement this board validator.")


################ PAWN STATE VALUATORS ################

class IsPassedPawn(BoardTabularEvaluator):
    def evaluate(self, board: jaxtyping[t.Tensor, 'batch 8 8 1']) -> jaxtyping[t.Tensor, 'batch 8 8 1']:
        out = t.zeros_like(board)
        if self.my:
            out |= board == PASSANT_PAWN
        if self.opponent:
            out |= board == (PASSANT_PAWN + OPPONENT_OFFSET)
        return out


class IsDoubledBlocked(BoardTabularEvaluator):
    def evaulate(self, board: jaxtyping[t.Tensor, 'batch 8 8 1']) -> jaxtyping[t.Tensor, 'batch 8 8 1']:
        n, _, _, _ = board.shape
        assert board.shape[1:] == (8, 8, 1)

        board = board.squeeze(-1)

        offset = 0 if self.my else OPPONENT_OFFSET
        look_for = [PAWN + offset, PASSANT_PAWN + offset]
        out = (board == look_for[0]) | (board == look_for[1])
        row_counts = t.sum(out, dim=0)
        assert row_counts.shape == (n, 8,)

        # take out columns that have only one pawn since it's not passed
        delete_columns = t.where(row_counts <= 1)
        assert len(delete_columns) == 2
        out[:, delete_columns] = False

        # take out the first bit of every column that has a bit
        ranks = t.cumsum(out, dim=0)
        files = t.arange(8)

        
        return out


class PawnInverseDistanceToPromotion(BoardTabularEvaluator):
    """
    A monotonic function that goes from 0 to 1 as the pawn gets closer to queening,
    per pawn. Shape like mask of pawns.
    """

    pass  # XXX


################ POSITIONAL + IDENTITIES + PIECE STATE VALUATORS ################
class IdentitiesStack(BoardTabularEvaluator):
    """
    A stack of identities of height 20. Basically a mask for where your pawns are, where your
    queen is, where your etc... is.
    """

    pass  # XXX

class PerIdentityReachableMaskStack(BoardTabularEvaluator):
    """
    A stack of identities of height 20. Basically a mask for where your pawns are, where your
    queen is, where your etc... is.
    """

    pass  # XXX


class EvenOddRank(BoardTabularEvaluator):
    """
    Mask for even or odd ranks.
    """

    def __init__(
        self,
        my: bool = False,
        opponent: bool = False,
        only_include_pieces: bool = False,
        default_value_for_empty: float = 0,
        even: bool = True,
    ) -> None:
        super().__init__(my, opponent, only_include_pieces, default_value_for_empty)
        self.even = even

    pass  # XXX


class DistanceToUnfriendlyKing(BoardTabularEvaluator):
    pass  # XXX


class DistanceToFriendlyKing(BoardTabularEvaluator):
    pass  # XXX


class Rank(BoardTabularEvaluator):
    """Rank = row"""

    pass  # XXX


class File(BoardTabularEvaluator):
    """File = column"""

    pass  # XXX


class DistanceFromCenter(BoardTabularEvaluator):
    """Manhattan distance from center."""

    pass  # XXX

class DevelopmentScore(BoardTabularEvaluator):
    """
    Score for development which varies based on pieces. Generally, it will be higher if pieces have
    moved relative to their origin or not.
    """
    pass

################ CLEVER PIECE STATE VALUATORS ################
class RookFiles(BoardTabularEvaluator):
    """
    Mask for where rook files are.
    """

    pass  # XXX


class BishopDiagonals(BoardTabularEvaluator):
    """
    Mask for where diagonals are for bishops.
    """

    pass  # XXX


class IsForking(BoardTabularEvaluator):
    """
    Whether this piece is forking two pieces, defined as attacking two pieces
    at once. This piece must not be under immediate attack, however, so that the
    fork is, in some sense, "real".
    """

    pass  # XXX


class IsPinned(BoardTabularEvaluator):
    """
    Whether the piece is pinned.
    """

    pass  # XXX


class MoveWouldDiscover(BoardTabularEvaluator):
    """
    Whether moving this piece would lead to a discovered check on the opponent of
    the mover.
    """

    pass  # XXX

class IsDoubleChecked(BoardTabularEvaluator):
    pass # XXX

class MoveWouldDoubleCheck(BoardTabularEvaluator):
    pass # XXX

class NumDefenders(BoardTabularEvaluator):
    pass  # XXX

# XXX come back to this! important!


################ ATTACK AND DEFENSE VALUATORS ################
# 1. Attacking
class IsAttacking_Source_PiecesCount_My(BoardTabularEvaluator):
    """How many pieces does this piece attack?"""

    pass  # XXX


# class IsAttacking_Source_PiecesCount_Opponent(BoardTabularEvaluator):
#     """Same for opponent."""

#     pass  # XXX


class IsAttacking_Source_Mask_My(BoardTabularEvaluator):
    """Whether this piece attacks anything."""

    pass  # XXX


# class IsAttacking_Source_Mask_Opponent(BoardTabularEvaluator):
#     """Same for oppponent."""

#     pass  # XXX


# 2. Attacked By
class IsAttackedBy_PiecesCount_My(BoardTabularEvaluator):
    """How many pieces attack this piece?"""

    pass  # XXX


# class IsAttackedBy_PiecesCount_Opponent(BoardTabularEvaluator):
#     """Same for opponent."""

#     pass  # XXX


class UnderAttack_Mask_My(BoardTabularEvaluator):
    """Whether this piece is under attack."""

    pass  # XXX


# class UnderAttack_Mask_Opponent(BoardTabularEvaluator):
#     """Same for opponent."""

#     pass  # XXX


class XraysThrough_Count_My(BoardTabularEvaluator):
    """
    Count, for this square, if this piece were to be gone, how many new pieces would be
    under attack that were not before? Basically, you can think of all pieces as having xrays
    that go through a single piece, and we are checking how many xrays pass through this
    piece. The more there are, the most this needs to be defended.
    """

    pass  # XXX


# class XraysThrough_Count_Opponent(BoardTabularEvaluator):
#     """
#     Same idea as above but for opponents.
#     """

#     pass  # XXX


# 3. Defended By
class IsDefendedBy_PiecesCount_My(BoardTabularEvaluator):
    """How many defend this piece (my)."""

    pass  # XXX


# class IsDefendedBy_PiecesCount_Opponent(BoardTabularEvaluator):
#     """How many defend this piece (opponent)."""

#     pass  # XXX


class IsDefended_Mask_My(BoardTabularEvaluator):
    """Whether or not this piece is defended (my)."""

    pass  # XXX


# class IsDefended_Mask_Opponent(BoardTabularEvaluator):
#     """Whether or not this piece is defended (opponent)."""

#     pass  # XXX


# 4. Reachability
class CanReachSquares_Count_My(BoardTabularEvaluator):
    """How many squares this piece can reach (my)."""

    pass  # XXX

class SafeReachableSqaures_Mask(BoardTabularEvaluator):
    """Which squares are reachable and not under attack."""
    pass
class SafeReachableSqaures_Count(BoardTabularEvaluator):
    """Count for the above."""
    pass

# class CanReachSquares_Count_Opponent(BoardTabularEvaluator):
#     """How many squares this piece can reach (opponent)."""

#     pass  # XXX


class CanReachXrays_Count(BoardTabularEvaluator):
    """
    How many pieces this piece can reach if the next piece that blocks
    its move is immediately removed.
    """

    def __init__(self, xray_steps: int) -> None:
        """
        Args:
            xray_steps (int): how many pieces to ignore to see how far this reaches.
        """
        self.xray_steps = xray_steps

    pass  # XXX


class ReachableSquares_Mask_My(BoardTabularEvaluator):
    """
    Which squares are reachable at all (by me). This is the same as mobility.
    """

    pass  # XXX


class ConcatBoardTabularEvaluator(BoardTabularEvaluator):
    def __init__(self, *valuators: List[BoardTabularEvaluator]) -> None:
        self.valuators = valuators

    def evaluate(
        self, board: jaxtyping.Float[t.Tensor, "8 8 1"]
    ) -> jaxtyping.Float[t.Tensor, "8 8 depth"]:
        return t.cat([val(board) for val in self.valuators], dim=-1)
