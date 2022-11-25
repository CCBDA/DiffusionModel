import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Grayscale,
    Lambda,
    Resize,
    ToTensor,
)

from data import DiffusionSet
from model import Diffusion, UNet

assert torch.cuda.is_available()


def getParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batchsize",
        default=32,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--steps",
        default=256,
        type=int,
        help="n step",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="load checkpoint from given path",
    )
    parser.add_argument(
        "--save",
        default=20,
        type=int,
        help="save every",
    )
    parser.add_argument(
        "--val",
        default=5,
        type=int,
        help="val every",
    )
    parser.add_argument(
        "--generate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="open generate mode, defulat to train mode",
    )
    return parser


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    IMG_SIZE = 32
    DEVICE = "cuda"
    BATCH_SIZE = args.batchsize
    N_STEPS = args.steps
    EPOCHS = args.epoch
    SAVE_EVERY = args.save
    VAL_EVERY = args.val
    trainTfs = Compose(
        [
            Resize((IMG_SIZE, IMG_SIZE)),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),
        ]
    )
    valTfs = Compose(
        [
            Resize((28, 28)),
            Grayscale(num_output_channels=3),
        ],
    )
    genTfs = Compose(
        [
            Resize((28, 28)),
            Grayscale(num_output_channels=3),
        ],
    )
    dataset = DiffusionSet(tfs=trainTfs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = UNet()
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt["model"])
        model.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        model.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.001)

    dfs = Diffusion(
        model=model,
        optimizer=optimizer,
        batchsize=BATCH_SIZE,
        n_steps=N_STEPS,
        device=DEVICE,
    )
    if args.generate:
        dfs.generateGrid(
            tfs=genTfs,
        )
        dfs.generate(
            total=10000,
            batchsize=50,
            tfs=genTfs,
        )
    else:
        dfs.train(
            epochs=EPOCHS,
            dataloader=dataloader,
            valEvery=VAL_EVERY,
            saveEvery=SAVE_EVERY,
            valtfs=valTfs,
        )
