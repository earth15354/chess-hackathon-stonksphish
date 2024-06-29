import torch as t
import torch.functional as F
import jaxtyping
from typing import List

################ PIECES ################
EMPTY = 0
PAWN, PASSANT_PAWN, UNMOVED_ROOK, MOVED_ROOK, KNIGHT = 1, 2, 3, 4, 5
LIGHT_BISHOP, DARK_BISHOP, QUEEN, UNMOVED_KING, MOVED_KING = 6, 7, 8, 9, 10
OPPONENT_OFFSET = 10

################ Valuator ################
class BoardValuator:
    """
    A board valuator is a class that really is meant to wrap a single function: valuate(...)
    which returns a per-cell value for the value of a board. It is meant, specifically, for
    SPACIAL valuations, based on where pieces are, etc...

    The right approach is to subclass this to create a bunch of valuators.
    """

    def valuate(
        self, board: jaxtyping.Float[t.Tensor, "8 8 1"]
    ) -> jaxtyping.Float[t.Tensor, "8 8 depth"]:
        raise NotImplementedError("Please implement this board validator.")

################ PAWN STATE VALUATORS ################

class IsPassedPawn(BoardValuator):
    pass # XXX
class IsDoubledBlocked(BoardValuator):
    pass # XXX
class DistanceToPromotionEvenOdd(BoardValuator):
    pass # XXX
class DistanceToPromotion(BoardValuator):
    pass # XXX


################ PIECE STATE VALUATORS ################
class DistanceToUnfriendlyKing(BoardValuator):
    pass # XXX
class DistanceToFriendlyKing(BoardValuator):
    pass # XXX

################ ATTACK AND DEFENSE VALUATORS ################
class IsAttackingPiecesCount(BoardValuator):
    pass # XXX
class IsAttackedByPiecesCount(BoardValuator):
    pass # XXX
class IsDefendedByPiecesCount(BoardValuator):
    pass # XXX
class CanReachSquaresCount(BoardValuator):
    pass # XXX
class XraysThroughCount(BoardValuator):
    """
    Count, for this square, if this piece were to be gone, how many new pieces would be
    under attack that were not before? Basically, you can think of all pieces as having xrays
    that go through a single piece, and we are checking how many xrays pass through this
    piece. The more there are, the most this needs to be defended.
    """
    pass # XXX

class ConcatBoardValidator:
    def __init__(self, *valuators: List[BoardValuator]) -> None:
        self.valuators = valuators

    def valute(
        self, board: jaxtyping.Float[t.Tensor, "8 8 1"]
    ) -> jaxtyping.Float[t.Tensor, "8 8 depth"]:
        return t.cat([val(board) for val in self.valuators], dim=-1)
