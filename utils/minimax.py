import torch as t
import numpy as np
import torch.nn as nn
from jaxtyping import Int, Float
import utils.chess_primitives as primitives
from typing import Optional, Dict, List
from models.convolutional import Model


class MiniMaxerV1:
    """
    A very simple MiniMax implementation with no performance optimization or any other effort, beyond
    the basic algorithm described here: https://en.wikipedia.org/wiki/Minimax#Minimax_algorithm_with_alternate_moves
    """

    def __init__(self, model: nn.Module, depth: int, **kwargs) -> None:
        self.model = model
        self.depth = depth
        self.n_states_explored = 0

    def generate_children(self, state: np.ndarray):
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
                for child in self.generate_children(board_state)
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
        assert len(children) == self.k

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


# TODO(Adriano) return to this, we might be able to get deeper search if we are able to
# optimize performance, which, in theory, we could do by maybe spreading load better or using
# more torch operations, etc...
#
# Batching is best, so we generate the tree, and then we calculate the values, and then we
# do the minimax calculation
# class Minimaxer:
#     def __init__(self, model: nn.Module, depth: int, roots: Int[t.Tensor, "batch 8 8"]) -> None:
#         assert roots.shape[1:] == (8, 8)
#         self.model = model
#         self.depth = depth
#         self.to_parent: Dict[int, Optional[int]] = {}
#         self.roots = {i : r for i, r in enumerate(self.roots)}
#         self.to_depth: Dict[int, int] = {i : 0 for i in self.roots.keys()} # Even depth = maximize
#         self.leaves: Dict[int, np.ndarray] = {}
#         for i, _ in enumerate(roots):
#             self.to_parent[i] = None

#     def trickle_down_phase(self) -> None:
#         first_idx = 0
#         final_idx = len(self.roots)
#         # for d in range(self.depth):
#         #     for parent_idx
#         #         children = generate_children(parent)
#         #         for child in children:
#         #             self.to_parent[]
#         #             new_queue.append(child)
#         #     queue = new_queue
#     def calculate_leaf_values_phase(self) -> None:
#         pass

#     def trickle_up_phase(self) -> None:
#         queue = []
#         for


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
