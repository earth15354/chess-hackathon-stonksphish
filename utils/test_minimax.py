import unittest
import numpy as np
import torch as t
import torch.nn as nn
from typing import List
from pathlib import Path
import click
import yaml
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())
from utils.minimax import MiniMaxerV1, MiniMaxedVanillaConvolutionalModel
from heuristics.utils import DEFAULT_STARTING_BOARD
from utils.chess_gameplay import Agent, play_game, play_tournament


class MockModel(nn.Module):
    def forward(self, x):
        return t.sum(x).float()


class TestMiniMaxerV1(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.depth = 2
        self.minimaxer = MiniMaxerV1(self.model, self.depth)

    def test_init(self):
        self.assertIsInstance(self.minimaxer.model, nn.Module)
        self.assertEqual(self.minimaxer.depth, 2)

    def test_generate_children(self):
        state = DEFAULT_STARTING_BOARD.numpy()
        children = self.minimaxer.generate_children(state)
        self.assertIsInstance(children, List)
        self.assertTrue(all(child.shape == (8, 8) for child in children))

    def test_minimax(self):
        state = DEFAULT_STARTING_BOARD.numpy()
        result = self.minimaxer.minimax(state, self.depth, True)
        self.assertIsInstance(result, float)

    def test_batch_minimax(self):
        batch_size = 3
        board_states = t.stack([DEFAULT_STARTING_BOARD for _ in range(batch_size)])
        results = self.minimaxer.batch_minimax(board_states)
        self.assertIsInstance(results, t.Tensor)
        self.assertEqual(results.shape, (batch_size,))


@click.command()
@click.option("--test-minimaxer-v1", is_flag=True)
@click.option("--play-minimaxed-vanilla-convolutional", is_flag=True)
@click.option("--model-config", type=str, default="model_config.yaml")
@click.option("--checkpoint", type=str, default="checkpoint.pt")
@click.option("--depth", type=int, default=4)
@click.option("--k", type=int, default=3)
def main(
    test_minimaxer_v1: bool,
    play_minimaxed_vanilla_convolutional: bool,
    model_config: str,
    checkpoint: str,
    depth: int,
    k: int,
):
    if test_minimaxer_v1:
        click.echo("Running MiniMaxerV1 tests")
        unittest.main(argv=["", "TestMiniMaxerV1"], exit=False)
    if play_minimaxed_vanilla_convolutional:
        click.echo("Playing MiniMaxed Vanilla Convolutional")
        # Load both models to be the same weights
        with open(model_config, "r") as f:
            kwargs = yaml.safe_load(f)
        if "depth" in kwargs:
            raise ValueError("Depth should not be in model config")
        kwargs["depth"] = depth
        kwargs["k"] = k
        kwargs["minimaxer_top_k"] = True  # Do this for perf. opt.
        vanilla_model1 = MiniMaxedVanillaConvolutionalModel(**kwargs)
        vanilla_model2 = MiniMaxedVanillaConvolutionalModel(**kwargs)
        state_dict = t.load(checkpoint)
        # names = sorted(state_dict["model"].keys())  # XXX
        # print("\n".join(names))  # XXX
        # print(len(names))
        # print("\n".join(sorted(vanilla_model1.state_dict().keys())))  # XXX
        # print(len(vanilla_model1.state_dict().keys()))  # XXX
        vanilla_model1.model.load_state_dict(state_dict["model"])
        vanilla_model1.eval()
        vanilla_model2.model.load_state_dict(state_dict["model"])
        vanilla_model2.eval()

        agents = [Agent(vanilla_model1), Agent(vanilla_model2)]
        # fmt: off
        game_result = play_game(
            table = 1,                                              # Used to send games to one tournament table or another
            agents = {'white': agents[0], 'black': agents[1]},      # We specify which agent is playing with which pieces
            max_moves = 50,                                         # Play until each agent has made up to 10 moves
            min_seconds_per_move = 0.1,                             # Artificially slow the game down for better watching
            verbose = True,                                         # Tell us what moves each agent makes and the outcome
            poseval = True                                          # Use stockfish to evaluate each position and fancy plot
        )
        # fmt: on
        click.echo(f"Game result:\n{game_result}")


if __name__ == "__main__":

    main()
