from __future__ import annotations
import torch
import unittest

from heuristics_v2 import evaluator_tabular, evaluator_vector
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

MAX_ALLOWABLE_TABLE_DEPTH = 512
MAX_ALLOWABLE_VECTOR_DEPTH = 512

from _test_utils import DEFAULT_STARTING_BOARD


class TestEvaluatePosition(unittest.TestCase):
    def test_evaluate_position(self):
        # Create a sample board
        board = DEFAULT_STARTING_BOARD.unsqueeze(0).float()
        # Symmetry and shape
        assert board.shape == (8, 8)
        # assert all(
        #     board[rank, file] == board[7 - rank, 7 - file] + OPPONENT_OFFSET
        #     for rank in range(4)
        #     for file in range(8)
        # )

        # Evaluate the position
        result_tabular = evaluator_tabular.evaluate_position(board)
        result_vector = evaluator_vector.evaluate_position(board)

        # Check that the results have the expected shape
        assert len(result_tabular.shape) == 4
        assert result_tabular.shape[0] == 1
        assert result_tabular.shape[1] <= MAX_ALLOWABLE_TABLE_DEPTH
        assert result_tabular.shape[-2:] == (8, 8)
        assert len(result_vector) == 2
        assert result_vector.shape[0] == 1
        assert result_vector.shape[1] <= MAX_ALLOWABLE_VECTOR_DEPTH

        # Check that all values are finite
        self.assertTrue(torch.all(torch.isfinite(result_tabular)))
        self.assertTrue(torch.all(torch.isfinite(result_vector)))


if __name__ == "__main__":
    unittest.main()
