from __future__ import annotations
import torch as t
import torch.nn as nn
from jaxtyping import Float
import einops
from typing import Tuple

"""
Chess architecture is meant to be able to learn many of the valuations that are done based on piece counts, etc...
"""

EMPTY = 0
PAWN, PASSANT_PAWN, UNMOVED_ROOK, MOVED_ROOK, KNIGHT = 1, 2, 3, 4, 5
LIGHT_BISHOP, DARK_BISHOP, QUEEN, UNMOVED_KING, MOVED_KING = 6, 7, 8, 9, 10
OPPONENT_OFFSET = 10
values = [
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
values += [v + OPPONENT_OFFSET for v in values]
values = t.Tensor(values).squeeze()
assert values.shape == (20,)

# TODO(adriano) ad some sort of attention

class LearnedValuation(nn.Module):
    def __init__(
        self,
        stdev: float = 1.0,
        mean: float = 0.0,
        num_valuations_per_piece_type: int = 10,
        latent_num_planes=800,
        num_internal_features: int = 100,
        num_external_features: int = 100,
        mlp_latent_dim: int = 512,
    ) -> None:
        self.num_planes = len(self.values) * num_valuations_per_piece_type
        assert latent_num_planes % 3 == 0 and latent_num_planes // 4 >= self.num_planes
        self.latent_num_planes = latent_num_planes

        # Value the piece types
        self.conv1d_valuations = nn.Conv1d(
            in_channels=len(self.values),
            out_channels=self.num_planes,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # Value the positions themselves (not the same as bias: value based on location AND
        # plane)
        self.bias_table = nn.Parameter(t.randn((8, 8, self.num_planes)) * stdev + mean)

        # Try to collect features w.r.t. chains of pawns, etc...
        self.conv3d = nn.Conv1d(
            in_channels=self.num_planes,
            out_channels=self.latent_num_planes // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.conv5d = nn.Conv1d(
            in_channels=self.num_planes,
            out_channels=self.latent_num_planes // 4,
            padding=2,
            kernel_size=5,
            stride=1,
            bias=False,
        )
        self.conv7d = nn.Conv1d(
            in_channels=self.num_planes,
            out_channels=self.latent_num_planes // 4,
            kernel_size=7,
            padding=3,
            stride=1,
            bias=False,
        )
        self.conv9d = nn.Conv1d(
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

        # self.multiplication_table = nn.Parameter(t.randn((8, 8, self.num_planes)) * stdev + mean)
        self.resnet = nn.Sequential([])
        self.mlp = nn.Sequential([])

    def preprocess_board(
        self, board: Float[t.Tensor, "batch height width"]
    ) -> Float[t.Tensor, "batch height width depth"]:
        batch, height, width = board.shape
        assert height == 8 and width == 8
        output = (
            einops.repeat(board, "batch height width repeat", repeat=len(values))
            == values
        )
        assert output.max() == 1 and output.min() == 0
        return output

    # TODO(Adriano) some sort of multimodal
    def forward(
        self,
        board: Float[t.Tensor, "batch height width"],
        value_estimators: Float[t.Tensor, "batch num_values"],
        moves_tokens: Float[t.Tensor, "batch moves_str"],
    ) -> Float[t.Tensor, "batch height width depth"]:
        assert board.shape[1:] == (8, 8)
        pieces_mask = self.preprocess_board(board)
        assert pieces_mask.shape[1:] == (8, 8, len(values))

        valuations = self.conv1d_valuations(pieces_mask)
        valuations += self.bias_table
        conv3d = self.conv3d(valuations)
        conv5d = self.conv5d(valuations)
        conv7d = self.conv7d(valuations)
        conv9d = self.conv9d(valuations)

        gathered_info = t.cat([conv3d, conv5d, conv7d, conv9d], dim=-1)
        assert gathered_info.shape == (board.shape[0], 8, 8, self.latent_num_planes)

        features = self.resnet(gathered_info)
        features = features.flatten()
        features = t.cat([features, value_estimators], dim=-1)

        latent = self.mlp(features)


        multiplication_table = einops.rearrange(
            self.multiplication_table, "h w -> 1 h w 1"
        )
        return board * multiplication_table
