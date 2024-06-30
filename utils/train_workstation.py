from __future__ import annotations
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.as_posix()))

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
import argparse
import os

# from model import Model
from models.convolutional import Model
from arch import LearnedValuation
from dataset import HDFDataset
from torch.utils.data import DataLoader
import socket
import yaml

from cycling_utils import (
    # InterruptableDistributedSampler,
    # MetricsTracker,
    atomic_torch_save,
)
from LAMB import Lamb
from jaxtyping import Float, Int
import torch.nn.functional as F
from arch import LearnedValuation, VALUES

timer.report("Completed imports")


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--load-dir", type=Path, default=None)
    parser.add_argument("--lr", type=float, required=True, default=None)
    return parser


def logish_transform(data):
    reflector = -1 * (data < 0).to(torch.int8)
    return reflector * torch.log(torch.abs(data) + 1)


def weighted_mse_loss(
    input: Float[torch.Tensor, "batch feats"],
    target: Float[torch.Tensor, "batch feats"],
    weight: Float[torch.Tensor, "feats"],
):
    batch_losses = torch.sum(weight * (input - target) ** 2, dim=-1)
    assert batch_losses.shape == (input.shape[0],)
    return torch.sum(batch_losses)  # TODO(Adriano) should we sum?


def get_actual_piece_counts(boards: Int[torch.Tensor, "batch 8 8"], device: str):
    piece_planes = LearnedValuation._preprocess_board(
        boards, VALUES.cuda(), device=device
    )
    assert piece_planes.shape == (boards.shape[0], 20, 8, 8)
    piece_counts = piece_planes.sum(dim=(2, 3))
    assert piece_counts.shape == (boards.shape[0], 20)
    return piece_counts


def main(args, timer):

    # dist.init_process_group("nccl")  # Expects RANK set in environment variable
    # rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    # world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    # args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    # args.is_master = rank == 0  # Master node for saving / reporting
    # torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'

    # if args.device_id:
    #     hostname = socket.gethostname()
    #     print("Hostname:", hostname)
    # timer.report("Setup for distributed training")

    args.save_chk_path = args.save_dir / "checkpoint.pt"
    if args.load_dir and not os.path.isfile(args.save_chk_path):
        # load from load path if one passed and save check path does not exist
        args.load_chk_path = args.load_dir
    else:
        # otherwise presume to save and load from the same place
        args.load_chk_path = args.save_chk_path
    args.save_chk_path.parent.mkdir(parents=True, exist_ok=True)
    timer.report("Validated checkpoint path")

    data_path = (
        Path(__file__).parent.parent / "data"
    ).as_posix()  # TODO(Adriano) please parameterize
    dataset = HDFDataset(data_path)
    timer.report("Loaded dataset to RAM")

    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        dataset, [0.8, 0.2], generator=random_generator
    )

    # train_sampler = InterruptableDistributedSampler(train_dataset)
    # test_sampler = InterruptableDistributedSampler(test_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    timer.report("Prepared dataloaders")

    model_config = yaml.safe_load(open(args.model_config))
    model = Model(**model_config)
    # model = LearnedValuation(**model_config)
    # model = model.to(args.device_id)
    model = model.to("cuda")
    # model = DDP(model, device_ids=[args.device_id])
    timer.report("Prepared model for distributed training")

    loss_weighting = torch.ones(21) * (1 / 42)
    loss_weighting[0] += 0.5
    loss_weighting = loss_weighting.to("cuda")
    assert torch.isclose(loss_weighting.sum(), torch.Tensor([1.0]).cuda())

    # loss_fn = nn.MSELoss(reduction="sum")
    loss_fn = lambda output, target: weighted_mse_loss(output, target, loss_weighting)
    optimizer = Lamb(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.9999),
        adam=False,
        weight_decay=0.0001,
        clamp_value=4,
    )
    # metrics = {"train": MetricsTracker(), "test": MetricsTracker()}

    if os.path.isfile(args.load_chk_path):
        if args.is_master:
            print(f"Loading checkpoint from {args.load_chk_path}")
        # checkpoint = torch.load(args.load_chk_path, map_location=f"cuda:{args.device_id}")
        checkpoint = torch.load(args.load_chk_path, map_location="cuda")

        # model.module.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        # test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        # metrics = checkpoint["metrics"]
        timer.report("Retrieved saved checkpoint")

    grad_accum_steps = 10
    save_steps = 10

    # for epoch in range(train_dataloader.sampler.epoch, 10000):
    for epoch in range(10_000):

        # with train_dataloader.sampler.in_epoch(epoch):

        timer.report(f"Training epoch {epoch}")
        train_steps_per_epoch = len(train_dataloader)
        optimizer.zero_grad()
        model.train()
        accum_loss = 0.0
        accum_value_loss = 0.0
        examples_seen = 0

        # for _moves, _turns, boards, evals in train_dataloader:
        for step, (_moves, _turns, boards, evals) in enumerate(train_dataloader):

            # Determine the current step
            # step = train_dataloader.sampler.progress // train_dataloader.batch_size
            is_last_step = (step + 1) == train_steps_per_epoch

            evals = logish_transform(evals)  # suspect this might help
            # boards, evals = boards.to(args.device_id), evals.to(args.device_id)
            boards, evals = boards.to("cuda"), evals.to("cuda")

            # Output
            scores = model(boards)
            assert scores.shape == (len(evals), 21)

            # Create the target
            evals = evals.unsqueeze(1)
            assert evals.shape == (len(evals), 1)

            pcounts = get_actual_piece_counts(boards, "cuda")
            assert pcounts.shape == (len(evals), 20)
            targets = torch.cat([evals, pcounts], dim=1)
            assert targets.shape == (len(evals), 21)

            loss = loss_fn(scores, targets)
            loss = loss / grad_accum_steps
            loss.backward()
            # train_dataloader.sampler.advance(len(evals))

            # top_eval_index = evals.argmax()
            # top1_score_indices = torch.topk(scores, 1).indices
            # top5_score_indices = torch.topk(scores, 5).indices

            # metrics["train"].update({
            #     "examples_seen": len(evals),
            #     "accum_loss": loss.item(),
            #     "top1_accuracy": 1 if top_eval_index in top1_score_indices else 0,
            #     "top5_accuracy": 1 if top_eval_index in top5_score_indices else 0
            # })
            accum_loss += loss.item()
            examples_seen += len(evals)
            accum_value_loss += (
                torch.square(scores[:, 0].detach() - targets[:, 0].detach()).sum().item()
            )

            if (step + 1) % grad_accum_steps == 0 or is_last_step:

                optimizer.step()
                optimizer.zero_grad()
                # metrics["train"].reduce()

                # if args.is_master:
                #     rpt = metrics["train"].local
                #     rpt_loss = rpt["accum_loss"]
                #     rpt_top1 = rpt["top1_accuracy"] / rpt["examples_seen"]
                #     rpt_top5 = rpt["top5_accuracy"] / rpt["examples_seen"]
                print(f"Step {step}, Loss {accum_loss / examples_seen:,.3f}")
                print(f"Step {step}, VALUE Loss {accum_value_loss / examples_seen:,.3f}")
                print(f"====")
                accum_loss = 0.0
                accum_value_loss = 0.0
                examples_seen = 0

                # metrics["train"].reset_local()

            # Saving
            # if ((step + 1) % save_steps == 0 or is_last_step) and args.is_master:
            if (step + 1) % save_steps == 0 or is_last_step:
                # Save checkpoint
                pth_str = args.save_chk_path
                pth = Path(pth_str)
                pth = pth.parent / f"{pth.name}-step-{step}{pth.suffix}"
                pth = pth.as_posix()
                atomic_torch_save(
                    {
                        # "model": model.module.state_dict(),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        # "train_sampler": train_dataloader.sampler.state_dict(),
                        # "test_sampler": test_dataloader.sampler.state_dict(),
                        # "metrics": metrics
                    },
                    pth,
                )

        # with test_dataloader.sampler.in_epoch(epoch):

        timer.report(f"Testing epoch {epoch}")
        test_steps_per_epoch = len(test_dataloader)
        model.eval()

        with torch.no_grad():
            # for _moves, _turns, boards, evals in test_dataloader:
            for step, (_moves, _turns, boards, evals) in enumerate(test_dataloader):

                # Determine the current step
                # step = test_dataloader.sampler.progress // test_dataloader.batch_size
                is_last_step = (step + 1) == test_steps_per_epoch

                evals = logish_transform(evals)  # suspect this might help
                # boards, evals = boards.to(args.device_id), evals.to(args.device_id)
                boards, evals = boards.to("cuda"), evals.to("cuda")
                scores = model(boards)

                evals = evals.unsqueeze(1)
                assert evals.shape == (len(evals), 1)

                pcounts = get_actual_piece_counts(boards, "cuda")
                assert pcounts.shape == (len(evals), 20)
                targets = torch.cat([evals, pcounts], dim=1)
                assert targets.device == scores.device
                assert targets.shape == (len(evals), 21)

                loss = loss_fn(scores, targets)

                loss = loss_fn(scores, evals)
                # test_dataloader.sampler.advance(len(evals))

                # top_eval_index = evals.argmax()
                # top1_score_indices = torch.topk(scores, 1).indices
                # top5_score_indices = torch.topk(scores, 5).indices

                # metrics["test"].update({
                #     "examples_seen": len(evals),
                #     "accum_loss": loss.item(),
                #     "top1_accuracy": 1 if top_eval_index in top1_score_indices else 0,
                #     "top5_accuracy": 1 if top_eval_index in top5_score_indices else 0
                #     })

                # Reporting
                if is_last_step:
                    # metrics["test"].reduce()

                    # if args.is_master:
                    #     rpt = metrics["test"].local
                    #     rpt_loss =rpt["accum_loss"] / rpt["examples_seen"]
                    #     rpt_top1 = rpt["top1_accuracy"] / rpt["examples_seen"]
                    #     rpt_top5 = rpt["top5_accuracy"] / rpt["examples_seen"]

                    print(
                        f"Epoch {epoch}, Loss {rpt_loss:,.3f}, Top1 {rpt_top1:,.3f}, Top5 {rpt_top5:,.3f}"
                    )

                    # metrics["test"].reset_local()

                # Saving
                # if ((step + 1) % save_steps == 0 or is_last_step) and args.is_master:
                if (step + 1) % save_steps == 0 or is_last_step:
                    pth_str = args.save_chk_path
                    pth = Path(pth_str)
                    pth = pth.parent / f"{pth.name}-step-{step}{pth.suffix}"
                    pth = pth.as_posix()
                    # Save checkpoint
                    atomic_torch_save(
                        {
                            # "model": model.module.state_dict(),
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            # "train_sampler": train_dataloader.sampler.state_dict(),
                            # "test_sampler": test_dataloader.sampler.state_dict(),
                            # "metrics": metrics
                        },
                        pth,
                    )


timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
