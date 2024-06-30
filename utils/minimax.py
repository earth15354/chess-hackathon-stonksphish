import torch as t
import numpy as np
import torch.nn as nn
from jaxtyping import Int, Float
import utils.chess_primitives as primitives
from typing import Optional, Dict, List
from models.convolutional import Model

from heuristics.utils import (
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


class PieceCounterModel(nn.Module):
    my_pieces = [
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
    ]
    enemy_pieces = [p + OPPONENT_OFFSET for p in my_pieces]

    def forward(self, x: Int[t.Tensor, "batch 8 8"]) -> Float[t.Tensor, "batch"]:
        mask = t.zeros_like(x)
        for p in self.my_pieces:
            mask += (x == p).int()
        for p in self.enemy_pieces:
            mask -= (x == p).int()
        return mask.sum(dim=-1).sum(dim=-1).float()


class MiniMaxerV1:
    """
    A very simple MiniMax implementation with no performance optimization or any other effort, beyond
    the basic algorithm described here: https://en.wikipedia.org/wiki/Minimax#Minimax_algorithm_with_alternate_moves
    """

    def __init__(self, model: nn.Module, depth: int, **kwargs) -> None:
        self.model = model
        self.depth = depth
        self.n_states_explored = 0

    @staticmethod
    def generate_children(state: np.ndarray):
        assert state.shape == (8, 8)
        # Outer: parents (states)
        # Inner: children (moves)
        assert isinstance(state, np.ndarray)
        children = [child for child in primitives.candidate_moves(state)]
        assert all(child.shape == (8, 8) for child in children)
        assert all(isinstance(child, np.ndarray) for child in children)
        return children

    def minimax(
        self,
        board_state: np.ndarray,
        depth_remaining: int,
        maximize: bool,
    ) -> float:
        self.n_states_explored += 1

        assert isinstance(board_state, np.ndarray)
        assert board_state.shape == (8, 8)
        if depth_remaining == 0:
            out = self.model(t.from_numpy(board_state)).item()
            assert isinstance(out, float)
            return out

        func = max if maximize else min
        return func(
            [
                self.minimax(child, depth_remaining - 1, not maximize)
                for child in MiniMaxerV1.generate_children(board_state)
            ]
        )

    def batch_minimax(
        self,
        board_states: Float[t.Tensor, "batch 8 8"],
    ) -> Float[t.Tensor, "batch"]:
        return t.Tensor(
            [
                self.minimax(board_state.numpy(), self.depth, True)
                for board_state in board_states
            ]
        )


class MiniMaxerV1TopK(MiniMaxerV1):
    """
    Small extension from above, but only checks the top k moves.
    """

    def __init__(self, model: nn.Module, depth: int, k: int) -> None:
        super().__init__(model, depth)
        self.k = k

    def minimax(
        self,
        board_state: np.ndarray,
        depth_remaining: int,
        maximize: bool,
    ) -> float:
        assert isinstance(board_state, np.ndarray)
        assert board_state.shape == (8, 8)
        if depth_remaining == 0:
            out = self.model(t.from_numpy(board_state)).item()
            assert isinstance(out, float)
            return out

        func = max if maximize else min
        children = self.generate_children(board_state)
        children_values = [
            (i, self.model(t.from_numpy(child)).item())
            for i, child in enumerate(children)
        ]
        children_values.sort(key=lambda x: x[1], reverse=maximize)
        children_values = children_values[: self.k]
        children = [children[i] for i, _ in children_values]
        assert all(isinstance(child, np.ndarray) for child in children)
        assert len(children) <= self.k

        if maximize:
            return func(
                [
                    self.minimax(child, depth_remaining - 1, not maximize)
                    for child in children
                ]
            )
        else:
            return func(
                [
                    self.minimax(child, depth_remaining - 1, not maximize)
                    for child in children
                ],
                key=lambda x: -x,
            )


class MinimaxerBatched:
    """
    A minimaxer implementation that aims to be maximally batched.
    """

    INFINITY = 2**62

    def __init__(
        self, model: nn.Module, depth: int, roots: Int[t.Tensor, "batch 8 8"]
    ) -> None:
        assert roots.shape[1:] == (8, 8)
        self.model = model
        self.depth = depth
        self.roots = roots
        self.parents = [None for _ in range(len(self.roots))]
        self.depths = [0 for _ in range(len(self.roots))]
        self.values = None
        self.end_exc = len(self.parents)
        self.leaves_boardstack = t.Tensor([])

    def trickle_down_phase(self) -> None:
        queue = [r.numpy() for r in self.roots]

        depth = 0
        n_visited = len(self.roots)
        while depth < self.depth:

            assert len(self.parents) == self.end_exc
            start_inc = self.end_exc - len(queue)
            # Ensure it's a rising sequence
            new_queue = []
            new_end_exc = self.end_exc
            for offset, state in enumerate(queue):
                children = MiniMaxerV1.generate_children(state)
                n_visited += len(children)
                new_queue += children
                self.depths += [
                    depth + 1 for _ in range(len(children))
                ]  # Add the to depths
                new_end_exc += len(children)
                self.parents.append(start_inc + offset)
            queue = new_queue
            self.end_exc = new_end_exc
            self.depth += 1
        assert len(queue) > 0
        self.leaves = queue
        self.leaves_boardstack = t.stack([t.from_numpy(l) for l in self.leaves])

        assert len(self.parents) == self.end_exc
        assert len(self.leaves) < len(self.parents)
        assert len(self.parents) == n_visited
        assert len(self.parents) == len(self.depths)
        # Ensure this is a monotonic array taht only goes up by 1 ever
        assert all(
            self.parents[i] <= self.parents[i + 1] for i in range(len(self.parents) - 1)
        )
        assert all(
            self.parents[i] + 1 >= self.parents[i + 1]
            for i in range(len(self.parents) - 1)
        )

        self.values = [None for _ in range(len(self.parents))]

    def tricke_up_phase(self) -> None:
        for value, depth, parent in reversed(
            zip(self.values, self.depths, self.parents)
        ):
            assert value is not None
            if parent is None:
                break

            if self.values[parent] is None:
                self.values[parent] = value
            else:
                func = max if depth % 2 == 0 else min
                self.values[parent] = func(
                    self.values[parent], value if depth % 2 == 0 else -value
                )
        assert all(v is not None and isinstance(v, float) for v in self.values)

    def minimax(self) -> None:
        # 1. Trickle down
        self.trickle_down_phase()

        # 2. Batch inference
        leaves_valuestack = self.model(self.leaves_boardstack)
        assert leaves_valuestack.shape == (len(self.leaves),)
        self.values[-len(leaves_valuestack) :] = leaves_valuestack.tolist()

        # 3. Trickle up
        self.tricke_up_phase()

        # 4. Extract values
        return t.Tensor(self.values[: self.len(self.roots)])


class MiniMaxedModule(nn.Module):
    def __init__(
        self,
        model: nn.Module = nn.Identity(),
        depth: int = 4,
        k: int = 4,
        minimaxer_top_k: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.depth = depth
        cls = MiniMaxerV1TopK if minimaxer_top_k else MiniMaxerV1
        self.minimaxer = cls(self.model, self.depth, k=k)

    def forward(self, x: Float[t.Tensor, "batch 8 8"]) -> Float[t.Tensor, "batch"]:
        return self.minimaxer.batch_minimax(x)


class MiniMaxedVanillaConvolutionalModel(MiniMaxedModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        if "depth" not in kwargs:
            raise ValueError("depth must be passed as a keyword argument")
        depth = kwargs.pop("depth")
        k = kwargs.pop("k")
        minimaxer_top_k = kwargs.pop("minimaxer_top_k")
        super().__init__(
            model=Model(**kwargs), depth=depth, k=k, minimaxer_top_k=minimaxer_top_k
        )


class MiniMaxedPieceCounterModel(MiniMaxedModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        if "depth" not in kwargs:
            raise ValueError("depth must be passed as a keyword argument")
        depth = kwargs.pop("depth")
        k = kwargs.pop("k")
        minimaxer_top_k = kwargs.pop("minimaxer_top_k")
        super().__init__(
            model=PieceCounterModel(), depth=depth, k=k, minimaxer_top_k=minimaxer_top_k
        )
