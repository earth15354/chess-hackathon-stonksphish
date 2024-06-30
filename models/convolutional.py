import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Int, Float
import utils.chess_primitives as primitives
from typing import List, Dict, Optional


class Residual(nn.Module):
    """The Residual block of ResNet models."""

    def __init__(self, outer_channels, inner_channels, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            outer_channels, inner_channels, kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            inner_channels, outer_channels, kernel_size=3, padding=1, stride=1
        )
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                outer_channels, outer_channels, kernel_size=1, stride=1
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.bn2 = nn.BatchNorm2d(outer_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class Model(nn.Module):
    """Convolutional Model"""

    def __init__(
        self,
        *args,
        ntoken: int = 21,
        embed_dim: int = 96,
        nlayers: int = 20,
        dropout: float = 0.5,
        nlayers_segments: int = 5,
    ):
        assert nlayers % nlayers_segments == 0
        super().__init__()
        self.model_type = "Transformer"
        self.src_mask = None
        self.ntoken = ntoken
        self.embed_dim = embed_dim

        self.input_emb = nn.Embedding(self.ntoken, self.embed_dim)
        self.convnet = nn.ParameterList(
            [
                nn.Sequential(
                    *[
                        Residual(self.embed_dim, 5 * self.embed_dim)
                        for _ in range(nlayers // nlayers_segments)
                    ]
                )
                for _ in range(nlayers_segments)
            ]
        )
        self.accumulator = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=8, padding=0, stride=1
        )
        self.decoder = nn.Linear(self.embed_dim, 1)

        self.count_pieces = nn.Linear(self.embed_dim, 20)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs, use_minimax: bool = False):  # (N, 8, 8)
        batch_size, _, _ = inputs.shape
        if len(inputs.shape) == 2:
            assert inputs.shape == (8, 8)
            inputs = inputs.unsqueeze(0)
        if use_minimax:
            raise NotImplementedError
        else:
            # print(inputs.shape)
            inputs = self.input_emb(inputs)  # (N, 8, 8, D) - this is nice
            # print(inputs.shape)
            inputs = torch.permute(inputs, (0, 3, 1, 2))  # (N, D, 8, 8)
            # print(inputs.shape)
            for i in range(len(self.convnet)):
                inputs = inputs + self.convnet[i](inputs)
            # print(inputs.shape)
            inputs = F.relu(self.accumulator(inputs).squeeze())
            piece_counts = self.count_pieces(inputs)
            assert piece_counts.shape == (batch_size, 20)
            # print(inputs.shape)
            scores = self.decoder(inputs).flatten()
            assert scores.shape == (batch_size,)
            scores = scores.unsqueeze(1)
            assert scores.shape == (batch_size, 1)
            output = torch.cat([scores, piece_counts], dim=1)
            assert output.shape == (batch_size, 21)
            # print(scores.shape)
            return output
