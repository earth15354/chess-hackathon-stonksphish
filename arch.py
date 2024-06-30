from __future__ import annotations
import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
import einops
from typing import Tuple

from models.convolutional import Residual

"""
Chess architecture is meant to be able to learn many of the valuations that are done based on piece counts, etc...
"""

EMPTY = 0
PAWN, PASSANT_PAWN, UNMOVED_ROOK, MOVED_ROOK, KNIGHT = 1, 2, 3, 4, 5
LIGHT_BISHOP, DARK_BISHOP, QUEEN, UNMOVED_KING, MOVED_KING = 6, 7, 8, 9, 10
OPPONENT_OFFSET = 10
VALUES = [
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
VALUES += [v + OPPONENT_OFFSET for v in VALUES]
VALUES = t.Tensor(VALUES).squeeze()
assert VALUES.shape == (20,)

# TODO(adriano) ad some sort of attention


class LearnedValuation(nn.Module):
    """A WIP model that is meant to try and do mostly residual, but also hybrid with transformers, operations
    to try and evaluate a chess board, while taking into account multiple heuristics and manually preprocessing
    information that might be hard to learn.

    Utility functions are also added to have loss functions that take into account representation learning
    (etc...).

    NOTE that the canonical representation we use is (chans x height x width).
    """

    def __init__(
        self,
        stdev: float = 1.0,
        mean: float = 0.0,
        num_valuations_per_piece_type: int = 10,
        latent_num_planes=800,
        num_internal_features: int = 100,
        num_external_features: int = 100,
        mlp_latent_dim: int = 512,
        n_resid_layers: int = 10,
        n_mlp_layers: int = 4,
        device: str = "cuda",
        values: t.Tensor = VALUES,
    ) -> None:
        super().__init__()
        self.num_planes = len(values) * num_valuations_per_piece_type
        assert latent_num_planes % 4 == 0 and latent_num_planes // 4 >= self.num_planes
        self.latent_num_planes = latent_num_planes

        # Value the piece types
        self.conv1x1_valuations = nn.Conv2d(
            in_channels=len(values),
            out_channels=self.num_planes,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # Value the positions themselves (not the same as bias: value based on location AND
        # plane)
        self.bias_table = nn.Parameter(t.randn((self.num_planes, 8, 8)) * stdev + mean)

        # Try to collect features w.r.t. chains of pawns, etc...
        self.conv3x3 = nn.Conv2d(
            in_channels=self.num_planes,
            out_channels=self.latent_num_planes // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=self.num_planes,
            out_channels=self.latent_num_planes // 4,
            padding=2,
            kernel_size=5,
            stride=1,
            bias=False,
        )
        self.conv7x7 = nn.Conv2d(
            in_channels=self.num_planes,
            out_channels=self.latent_num_planes // 4,
            kernel_size=7,
            padding=3,
            stride=1,
            bias=False,
        )
        self.conv9x9 = nn.Conv2d(
            in_channels=self.num_planes,
            out_channels=self.latent_num_planes // 4,
            kernel_size=9,
            padding=4,
            stride=1,
            bias=False,
        )

        self.num_internal_features = num_internal_features
        self.num_external_features = num_external_features
        self.mlp_latent_dim = mlp_latent_dim
        self.n_resid_layers = n_resid_layers
        self.n_mlp_layers = n_mlp_layers

        # self.multiplication_table = nn.Parameter(t.randn((8, 8, self.num_planes)) * stdev + mean)
        self.resnet = self.convnet = nn.Sequential(
            *[
                Residual(self.latent_num_planes, 4 * self.latent_num_planes)
                for _ in range(self.n_resid_layers)
            ]
        )
        self.projector_to_mlp = nn.Conv2d(
            in_channels=self.latent_num_planes,
            out_channels=self.mlp_latent_dim,
            kernel_size=8,
            stride=1,
        )
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.BatchNorm1d(self.mlp_latent_dim),
                    nn.Linear(self.mlp_latent_dim, 4 * self.mlp_latent_dim),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(4 * self.mlp_latent_dim, self.mlp_latent_dim),
                )
                for _ in range(self.n_mlp_layers)
            ]
        )
        self.projector_to_value = nn.Linear(self.mlp_latent_dim, 1)

        self.device = device

        self.values = values.to(self.device)

    def preprocess_board(
        self, board: Int[t.Tensor, "batch height width"]
    ) -> Float[t.Tensor, "batch depth height width"]:
        batch, height, width = board.shape
        assert height == 8 and width == 8
        output = (
            einops.repeat(board, "b h w -> b h w r", r=len(self.values)) == self.values
        ).float()
        output_with_chans = einops.rearrange(output, "b h w r -> b r h w ")
        assert output_with_chans.max() == 1 and output_with_chans.min() == 0
        return output_with_chans.to(self.device)

    # TODO(Adriano) some sort of multimodal
    def forward(
        self,
        board: Int[t.Tensor, "batch height width"],
        # value_estimators: Float[t.Tensor, "batch num_values"],
        # moves_tokens: Float[t.Tensor, "batch moves_str"],
    ) -> Float[t.Tensor, "batch height width depth"]:
        assert board.shape[1:] == (8, 8)
        assert board.dtype == t.int32 or board.dtype == t.int64
        batch, _, _ = board.shape

        pieces_mask = self.preprocess_board(board)
        assert pieces_mask.shape == (
            batch,
            len(self.values),
            8,
            8,
        ), f"{pieces_mask.shape}"

        valuations = self.conv1x1_valuations(pieces_mask)
        valuations += self.bias_table
        conv3x3 = self.conv3x3(valuations)
        conv5x5 = self.conv5x5(valuations)
        conv7x7 = self.conv7x7(valuations)
        conv9x9 = self.conv9x9(valuations)

        gathered_info = t.cat([conv3x3, conv5x5, conv7x7, conv9x9], dim=1)
        assert gathered_info.shape == (
            batch,
            self.latent_num_planes,
            8,
            8,
        ), f"{gathered_info.shape}"

        features = self.resnet(gathered_info)
        projected = self.projector_to_mlp(features)
        projected = einops.rearrange(
            projected, "b c h w -> b (c h w)", c=self.mlp_latent_dim, h=1, w=1
        )
        assert projected.shape == (batch, self.mlp_latent_dim), f"{projected.shape}"
        # features = t.cat([features, value_estimators], dim=-1)

        latent = self.mlp(projected)
        value = self.projector_to_value(latent)

        return value
