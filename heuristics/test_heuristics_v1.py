import torch
import unittest

try:
    from heuristics.heuristics_v1 import evaluate_position
    from heuristics.heuristics_v1 import (
        UNMOVED_ROOK,
        KNIGHT,
        LIGHT_BISHOP,
        QUEEN,
        UNMOVED_KING,
        DARK_BISHOP,
        PAWN,
        EMPTY,
        OPPONENT_OFFSET,
    )
except (ModuleNotFoundError, ImportError):
    from heuristics_v1 import evaluate_position
    from heuristics_v1 import (
        UNMOVED_ROOK,
        KNIGHT,
        LIGHT_BISHOP,
        QUEEN,
        UNMOVED_KING,
        DARK_BISHOP,
        PAWN,
        EMPTY,
        OPPONENT_OFFSET,
    )


class TestEvaluatePosition(unittest.TestCase):
    def test_evaluate_position(self):
        # Create a sample board
        # fmt: off
        board = torch.tensor([
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
        # Symmetry and shape
        assert board.shape == (8, 8)
        # assert all(
        #     board[rank, file] == board[7 - rank, 7 - file] + OPPONENT_OFFSET
        #     for rank in range(4)
        #     for file in range(8)
        # )

        # Evaluate the position
        result = evaluate_position(board)

        # Check that the result has the expected shape
        self.assertEqual(result.shape, (14,))  # 14 features are calculated

        # Check that all values are finite
        self.assertTrue(torch.all(torch.isfinite(result)))

if __name__ == "__main__":
    unittest.main()
