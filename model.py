import torch
import torch.nn as nn
import torch.nn.functional as F
import heuristics.heuristics_v1 as hv1
class Residual(nn.Module):
    """The Residual block of ResNet models."""
    def __init__(self, outer_channels, inner_channels, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(outer_channels, inner_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(inner_channels, outer_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(outer_channels, outer_channels, kernel_size=1, stride=1)
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

HEURISTICS_SIZE = 14

class Model(nn.Module):
    """Convolutional Model"""

    def __init__(self, ntoken, embed_dim, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ntoken = ntoken
        self.embed_dim = embed_dim

        self.input_emb = nn.Embedding(self.ntoken, self.embed_dim)
        self.convnet = nn.Sequential(*[Residual(self.embed_dim, 5 * self.embed_dim) for _ in range(nlayers)])
        self.accumulator = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=8, padding=0, stride=1)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim + HEURISTICS_SIZE, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, 1))
        
        
    def forward(self, og_inputs): # (N, 8, 8)
        n, r, c = og_inputs.shape
        assert r == 8 and c == r
        # print(inputs.shape)
        inputs = self.input_emb(og_inputs) # (N, 8, 8, D) - this is nice
        # print(inputs.shape)
        inputs = torch.permute(inputs, (0, 3, 1, 2))
        # print(inputs.shape)
        inputs = self.convnet(inputs)
        # print(inputs.shape)
        inputs = F.relu(self.accumulator(inputs).squeeze())
        assert len(inputs.shape) == 1
        heuristics = torch.tensor([hv1.evaluate_position(i) for i in og_inputs])
        assert heuristics.shape == (n, HEURISTICS_SIZE)
        assert inputs.shape == (n, self.embed_dim)
        concattenated = torch.cat([heuristics, inputs], dim=1)
        assert concattenated.shape == (n, HEURISTICS_SIZE + self.embed_dim)
        # print(inputs.shape)
        scores = self.decoder(concattenated).flatten()
        assert scores.shape == (n,)
        # print(scores.shape)
        return scores