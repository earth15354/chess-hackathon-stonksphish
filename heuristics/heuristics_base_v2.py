from __future__ import annotations

EMPTY = 0
PAWN, PASSANT_PAWN, UNMOVED_ROOK, MOVED_ROOK, KNIGHT = 1, 2, 3, 4, 5
LIGHT_BISHOP, DARK_BISHOP, QUEEN, UNMOVED_KING, MOVED_KING = 6, 7, 8, 9, 10
OPPONENT_OFFSET = 10


class BaseEvaluator(object):
    def __init__(
        self,
        my: bool = False,
        opponent: bool = False,
        only_include_pieces: bool = False,
        default_value_for_empty: float = 0.0,
    ) -> None:
        """
        Args:
            my (bool, optional): whether or not to include my pieces. Defaults to False.
            opponent (bool, optional): whether or not to include opp. pieces. Defaults to False.
            only_include_pieces (bool, optional): whether or not, if applicable,
                to include squares that are not for pieces. Defaults to False.
            default_value_for_empty (float, optional): the default value for empty
                squares (i.e. what to put in places where there were not pieces if
                you passed only_include_pieces=True). Defaults to 0.0.
        """
        self.my = my
        self.opponent = opponent
        self.only_include_pieces = only_include_pieces
        self.default_value_for_empty = default_value_for_empty
