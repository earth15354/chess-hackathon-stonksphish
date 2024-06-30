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
    from heuristics.utils import DEFAULT_STARTING_BOARD
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
    from utils import DEFAULT_STARTING_BOARD


class TestEvaluatePosition(unittest.TestCase):
    def test_evaluate_position(self):
        # Create a sample board
        board = DEFAULT_STARTING_BOARD
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
