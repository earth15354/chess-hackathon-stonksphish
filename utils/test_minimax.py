import unittest
import numpy as np
import time
import torch as t
import torch.nn as nn
from typing import List
from pathlib import Path
import click
import yaml
import sys
import tqdm

sys.path.append(Path(__file__).parent.parent.as_posix())
from utils.minimax import (
    MiniMaxerV1,
    MiniMaxedVanillaConvolutionalModel,
    MiniMaxedPieceCounterModel,
    MinimaxerBatched,
)
from models.convolutional import Model as VanillaModel
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


class TestMinimaxerBatched(unittest.TestCase):
    def test_run(self):
        # Mock the model (we don't need it for this test)
        model = MockModel()

        # Create a sample root state
        roots = t.stack([DEFAULT_STARTING_BOARD for _ in range(2)], dim=0)
        assert roots.shape == (2, 8, 8)

        # Initialize MinimaxerBatched
        depth = 3
        minimaxer = MinimaxerBatched(model, roots, depth)

        # Run the trickle_down_phase
        minimaxer.trickle_down_phase()


@click.command()
@click.option("--test-minimaxer-v1", is_flag=True)
@click.option("--play", is_flag=True)
@click.option(
    "--play-model", type=click.Choice(["vanilla-convolutional", "piece-count"])
)
@click.option("--model-config", type=str, default="model_config.yaml")
@click.option("--checkpoint", type=str, default="checkpoint.pt")
@click.option("--depth", type=int, default=4)
@click.option("--k", type=int, default=3)
@click.option("--benchmark-minimaxer-v1", is_flag=True)
@click.option("--benchmark-vanilla-convolutional", is_flag=True)
@click.option("--device", type=str, default="cpu")
@click.option("--test-minimaxer-batched", is_flag=True)
@click.option("--benchmark-batched-inference", is_flag=True)
@click.option("--benchmark-batched-minimax", is_flag=True)
def main(
    test_minimaxer_v1: bool,
    play: bool,
    play_model: str,
    model_config: str,
    checkpoint: str,
    depth: int,
    k: int,
    benchmark_minimaxer_v1: bool,
    benchmark_vanilla_convolutional: bool,
    device: str,
    test_minimaxer_batched: bool,
    benchmark_batched_inference: bool,
    benchmark_batched_minimax: bool,
):
    if test_minimaxer_v1:
        click.echo("Running MiniMaxerV1 tests")
        unittest.main(argv=["", "TestMiniMaxerV1"], exit=False)
    if play:
        click.echo("Playing MiniMaxed Vanilla Convolutional")
        # Load both models to be the same weights
        with open(model_config, "r") as f:
            kwargs = yaml.safe_load(f)
        if "depth" in kwargs:
            raise ValueError("Depth should not be in model config")
        kwargs["depth"] = depth
        kwargs["k"] = k
        kwargs["minimaxer_top_k"] = True  # Do this for perf. opt.
        vanilla_model1 = (
            MiniMaxedVanillaConvolutionalModel(**kwargs).to(device)
            if play_model == "vanilla-convolutional"
            else MiniMaxedPieceCounterModel(**kwargs).to(device)
        )
        vanilla_model2 = (
            MiniMaxedVanillaConvolutionalModel(**kwargs).to(device)
            if play_model == "vanilla-convolutional"
            else MiniMaxedPieceCounterModel(**kwargs).to(device)
        )
        if play_model == "vanilla-convolutional":
            state_dict = t.load(checkpoint)
            # names = sorted(state_dict["model"].keys())  # XXX
            # print("\n".join(names))  # XXX
            # print(len(names))
            # print("\n".join(sorted(vanilla_model1.state_dict().keys())))  # XXX
            # print(len(vanilla_model1.state_dict().keys()))  # XXX
            vanilla_model1.model.load_state_dict(state_dict["model"])
            vanilla_model2.model.load_state_dict(state_dict["model"])
        vanilla_model1.eval()
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
    if benchmark_minimaxer_v1:
        click.echo("Benchmarking MiniMaxerV1")
        model = MockModel()  # Very fast, should not really add much to the time
        for depth in range(1, 10):
            click.echo(f"Depth: {depth}...")
            t_start = time.time()
            minimaxer = MiniMaxerV1(model, depth)
            minimaxer.batch_minimax([DEFAULT_STARTING_BOARD.to("cpu")])
            t_end = time.time()
            click.echo(
                f"Depth: {depth}, Time: {t_end - t_start}, explored {minimaxer.n_states_explored} states"
            )
    if benchmark_vanilla_convolutional:
        with open(model_config, "r") as f:
            kwargs = yaml.safe_load(f)
        vanilla_model1 = VanillaModel(**kwargs).to(device)
        state_dict = t.load(checkpoint)
        vanilla_model1.load_state_dict(state_dict["model"])
        vanilla_model1.eval()
        input = DEFAULT_STARTING_BOARD.unsqueeze(0).to(device)
        for model, message in zip(
            [MockModel(), vanilla_model1], ["Baseline", "Vanilla Convolutional"]
        ):
            click.echo("=" * 32 + f" {message} " + "=" * 32)
            for i in range(5):
                time_start = time.time()
                for _ in tqdm.trange(100):
                    vanilla_model1(input)
                time_end = time.time()
                avg_time_taken = (time_end - time_start) / 100
                click.echo(f"Trial {i+1}, Average time taken (/1000): {avg_time_taken}")
            click.echo("=" * 128)

    if test_minimaxer_batched:
        click.echo("Running MinimaxerBatched tests")
        unittest.main(argv=["", "TestMinimaxerBatched"], exit=False)

    if benchmark_batched_inference:
        click.echo("Benchmarking batched inference")
        with open(model_config, "r") as f:
            kwargs = yaml.safe_load(f)
        vanilla_model1 = VanillaModel(**kwargs).to(device)
        state_dict = t.load(checkpoint)
        vanilla_model1.load_state_dict(state_dict["model"])
        vanilla_model1.eval()
        vanilla_model1 = vanilla_model1.to(device)
        for i in range(5):
            time_start = time.time()
            inputs = [DEFAULT_STARTING_BOARD for _ in range(1000)]
            inputs = t.stack(inputs, dim=0).to(device)
            vanilla_model1(inputs)
            assert inputs.shape == (1000, 8, 8)
            time_end = time.time()
            avg_time_taken = (time_end - time_start) / 1000
            click.echo(f"Trial {i+1}, Average time taken (/1000): {avg_time_taken}")

    if benchmark_batched_minimax:
        click.echo("Benchmarking batched minimax")
        with open(model_config, "r") as f:
            kwargs = yaml.safe_load(f)
        vanilla_model1 = VanillaModel(**kwargs).to(device)
        state_dict = t.load(checkpoint)
        vanilla_model1.load_state_dict(state_dict["model"])
        vanilla_model1.eval()
        vanilla_model1 = vanilla_model1.to(device)
        for depth in range(1, 3):
            for i in range(3):
                time_start = time.time()
                roots = t.stack([DEFAULT_STARTING_BOARD for _ in range(32)], dim=0)
                assert roots.shape == (32, 8, 8), roots.shape
                minimaxer = MinimaxerBatched(vanilla_model1, roots, depth)
                minimaxer.minimax()
                time_end = time.time()
                avg_time_taken = (time_end - time_start) / 32
                click.echo(
                    f"Trial {i+1}, depth {depth}: Average time taken (/32 batch size): {avg_time_taken}"
                )


if __name__ == "__main__":

    main()
