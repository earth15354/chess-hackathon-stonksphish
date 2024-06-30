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

MY_PIECES = [
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
ENEMY_PIECES = [p + OPPONENT_OFFSET for p in MY_PIECES]


def count_pieces_torch(x: Int[t.Tensor, "batch 8 8"]) -> Float[t.Tensor, "batch"]:
    mask = t.zeros_like(x)
    for p in MY_PIECES:
        mask += (x == p).int()
    for p in ENEMY_PIECES:
        mask -= (x == p).int()
    return mask.sum(dim=-1).sum(dim=-1)


def count_my_pieces_np(x: np.ndarray) -> float:
    return ((x >= 1) & (x <= 10)).sum()


class PieceCounterModel(nn.Module):

    def forward(self, x: Int[t.Tensor, "batch 8 8"]) -> Float[t.Tensor, "batch"]:
        count_pieces_torch(x).float()


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
        self,
        model: nn.Module,
        roots: Int[t.Tensor, "batch 8 8"],
        depth: int = 4,
        device: str = "cpu",
    ) -> None:
        assert roots.shape[1:] == (8, 8)
        self.model = model
        self.depth = depth
        self.roots = roots
        self.ids = 0
        self.parent_ids: Dict[int, Optional[int]] = {}
        self.leaf_id2leaf: Dict[int, np.ndarray] = {}
        self.id2value: Dict[int, float] = {}
        self.id2depth: Dict[int, int] = {}
        self.device = device

    def _trickle_down(
        self,
        board_state: np.ndarray,
        depth_remaining: int,
        parent_id: Optional[int],
    ) -> float:
        assert isinstance(board_state, np.ndarray)
        assert board_state.shape == (8, 8)

        # Grab an id
        assigned_id = self.ids
        self.ids += 1

        # Point to parent
        self.parent_ids[assigned_id] = parent_id

        if depth_remaining == 0:
            self.leaf_id2leaf[assigned_id] = board_state
        else:
            # Leaves don't need to know their depth (check trickle up)
            self.id2depth[assigned_id] = self.depth - depth_remaining
            for child in MiniMaxerV1.generate_children(board_state):
                self._trickle_down(
                    board_state=child,
                    depth_remaining=depth_remaining - 1,
                    parent_id=assigned_id,
                )

    def trickle_down_phase(self) -> None:
        for _, root in enumerate(self.roots):
            self._trickle_down(
                board_state=root.numpy(), depth_remaining=self.depth, parent_id=None
            )

    def batch_infer_phase(self) -> None:
        # print(self.leaf_id2leaf.keys()) # DEBUG
        # print(set(range(self.ids - len(self.leaf_id2leaf), self.ids))) # DEBUG
        # assert set(self.leaf_id2leaf.keys()) == set(range(self.ids - len(self.leaf_id2leaf), self.ids))

        leaf_idxs = sorted(self.leaf_id2leaf.keys())

        # Get the leaves
        # Support None so we can have precomputed (heuristic) values
        leaves = t.stack(
            [
                (
                    t.Tensor(self.leaf_id2leaf[i]).long().to(self.device)
                    if self.leaf_id2leaf[i] is not None
                    else None
                )
                for i in leaf_idxs
            ],
            dim=0,
        )
        assert leaves.shape == (len(leaf_idxs), 8, 8)
        leaves_valuestack = self.model(leaves)
        for i, value in enumerate(leaves_valuestack):
            idx = leaf_idxs[i]
            if value is not None:
                assert idx not in self.id2value
                self.id2value[idx] = value
            else:
                assert idx in self.id2value
                assert isinstance(self.id2value[idx], float)

    # Return what a BAD move is
    def __default_value_at(self, depth: int) -> float:
        return -self.INFINITY if depth % 2 == 0 else self.INFINITY

    def trickle_up_phase(self) -> None:
        for i in range(self.ids - 1, -1, -1):
            assert i in self.id2value
            if self.parent_ids[i] is None:
                # Does not apply because DFS instead of BFS
                # assert all(self.parent_ids[j] is None for j in range(i))
                continue
            parent_id = self.parent_ids[i]
            parent_depth = self.id2depth[parent_id]
            parent_value = self.id2value.get(
                parent_id, self.__default_value_at(parent_depth)
            )

            child_value = (
                self.id2value[i] if parent_depth % 2 == 0 else -self.id2value[i]
            )
            self.id2value[parent_id] = max(parent_value, child_value)

    def minimax(self) -> None:
        # 1. Trickle down
        self.trickle_down_phase()

        # 2. Batch inference
        self.batch_infer_phase()

        # 3. Trickle up
        self.trickle_up_phase()

        # 4. Extract values
        out = t.Tensor([self.id2value[i] for i in range(len(self.roots))])
        assert out.shape == (len(self.roots),)
        return out


class MinimaxerBatchedAvoidPieceLosingMoves(MinimaxerBatched):
    """
    Does minimax batched but with one key simple functionality that is that it automatically
    turns moves that lose pieces into LEAVES with -infinity score so that they won't be made.
    """

    def _trickle_down(
        self,
        board_state: np.ndarray,
        depth_remaining: int,
        parent_id: Optional[int],
        parent_piece_count: int,
    ) -> float:
        assert isinstance(board_state, np.ndarray)
        assert board_state.shape == (8, 8)

        # Grab an id
        assigned_id = self.ids
        self.ids += 1

        # Point to parent
        self.parent_ids[assigned_id] = parent_id

        if depth_remaining == 0:
            self.leaf_id2leaf[assigned_id] = board_state
        else:
            # Leaves don't need to know their depth (check trickle up)
            self.id2depth[assigned_id] = self.depth - depth_remaining
            for child in MiniMaxerV1.generate_children(board_state):
                child_piece_count = count_my_pieces_np(child)
                if child_piece_count < parent_piece_count:
                    worst_value = (
                        -self.INFINITY if depth_remaining % 2 == 0 else self.INFINITY
                    )
                    self.child_assigned_id = self.ids
                    self.ids += 1
                    self.leaf_id2leaf[self.child_assigned_id] = None
                    self.id2value[self.child_assigned_id] = worst_value
                else:
                    self._trickle_down(
                        board_state=child,
                        depth_remaining=depth_remaining - 1,
                        parent_id=assigned_id,
                        parent_piece_count=child_piece_count,
                    )

    def trickle_down_phase(self) -> None:
        for _, root in enumerate(self.roots):
            self._trickle_down(
                board_state=root.numpy(),
                depth_remaining=self.depth,
                parent_id=None,
                parent_piece_count=16,
            )


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
