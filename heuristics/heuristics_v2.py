from __future__ import annotations

from heuristics_tabular_v2 import (
    ConcatBoardTabularEvaluator,
    IsDefended_Mask_My,
    CanReachSquares_Count_My,
    CanReachXrays_Count,
    ReachableSquares_Mask_My,
)
from heuristics_vector_v2 import ConcatBoardVectorEvaluator, InverseMeanDistanceFromKing

evaluator_tabular = ConcatBoardTabularEvaluator(
    IsDefended_Mask_My(),
    CanReachSquares_Count_My(),
    CanReachXrays_Count(1),
    ReachableSquares_Mask_My(),
)

evaluator_vector = ConcatBoardVectorEvaluator(
    InverseMeanDistanceFromKing(),
)
