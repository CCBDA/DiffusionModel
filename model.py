import os
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64, 16)
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128, 8)
        self.down3 = Down(128, 128)
        self.sa3 = SelfAttention(128, 4)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        self.sa4 = SelfAttention(64, 8)
        self.up2 = Up(128, 32)
        self.sa5 = SelfAttention(32, 16)
        self.up3 = Up(64, 32)
        self.sa6 = SelfAttention(32, 32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64, 16)
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128, 8)
        self.down3 = Down(128, 128)
        self.sa3 = SelfAttention(128, 4)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        self.sa4 = SelfAttention(64, 8)
        self.up2 = Up(128, 32)
        self.sa5 = SelfAttention(32, 16)
        self.up3 = Up(64, 32)
        self.sa6 = SelfAttention(32, 32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class Diffusion:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batchsize: int,
        n_steps: int = 300,
        device="cuda",
    ) -> None:
        self.n_steps = n_steps
        self.beta = torch.linspace(0.0001, 0.04, n_steps, device=device)
        self.alpha_bar = torch.cumprod(1.0 - self.beta, dim=0)
        self.alpha = 1 - self.beta
        self.batchsize = batchsize
        self.device = device
        self.model = model
        self.optimizer = optimizer

    def gather(self, consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather consts for $t$ and reshape to feature map shape"""
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)

    def noise_x_by_t_step(
        self,
        x_clear: torch.Tensor,
        t_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.gather(self.alpha_bar, t_step) ** 0.5 * x_clear
        var = 1.0 - self.gather(self.alpha_bar, t_step)
        noise = torch.randn_like(x_clear)
        x_noised = mean + (var**0.5) * noise
        return x_noised, noise

    def compute_loss(self, x_clear: torch.Tensor) -> torch.Tensor:
        t_step = torch.randint(
            0, self.n_steps, (self.batchsize,), dtype=torch.long, device=self.device
        )
        x_noised, noise = self.noise_x_by_t_step(x_clear, t_step)
        pred_noise = self.model(x_noised, t_step)
        loss = F.mse_loss(noise, pred_noise)
        return loss

    def p_xt(
        self,
        xt: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        alpha_t = self.gather(self.alpha, t)
        alpha_bar_t = self.gather(self.alpha_bar, t)
        eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** 0.5
        mean = 1 / (alpha_t**0.5) * (xt - eps_coef * noise)  # Note minus sign
        var = self.gather(self.beta, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5) * eps

    def save(self, epoch: int, folder="checkpoint"):
        os.makedirs(folder, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "{}/model_{}.ckpt".format(folder, epoch),
        )

    def train(
        self,
        epochs: int,
        dataloader: DataLoader,
        valEvery: int = 5,
        saveEvery: int = 50,
        valtfs: transforms.Compose = None,
        valfolder="validate",
    ):
        shutil.rmtree(valfolder)
        os.makedirs(valfolder)
        print("Start training")
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch in tqdm(dataloader):
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch}  Loss: {loss.item()} ")
            if epoch % valEvery == 0:
                self.validate(epoch, tfs=valtfs),
            if epoch % saveEvery == 0:
                self.save(epoch)

    def validate(
        self,
        epoch: int,
        dim: int = 32,
        device: str = "cuda",
        numImages: int = 8,
        tfs: transforms.Compose = None,
        folder="validate",
    ):
        self.model.eval()
        with torch.no_grad():
            img = torch.randn((1, 3, dim, dim), device=device)
            plt.figure(figsize=(15, 15))
            plt.axis("off")
            stepsize = self.n_steps // numImages
            result = []
            for i in range(0, self.n_steps):
                t = torch.tensor(self.n_steps - i - 1.0, device="cuda").long()
                pred_noise = self.model(img, t.unsqueeze(0))
                img = self.p_xt(img, pred_noise, t.unsqueeze(0))
                if i % stepsize == 0:
                    if len(img.shape) == 4:
                        output = img[0, :, :, :]
                    if tfs is not None:
                        output = tfs(output)
                    result.append(output)
            result = torch.stack(result, dim=0)
            save_image(result, "{}/{}.png".format(folder, epoch))

    def generateGrid(
        self,
        tfs: transforms.Compose = None,
    ):
        with torch.no_grad():
            x = torch.randn(8, 3, 32, 32).cuda()
            stepsize = self.n_steps // 8
            result = []
            for i in range(self.n_steps):
                t = torch.tensor(self.n_steps - i - 1.0, device="cuda").long()
                pred_noise = self.model(x, t.unsqueeze(0))
                x = self.p_xt(x, pred_noise, t.unsqueeze(0))
                if i % stepsize == 0:
                    for img in x:
                        output = tfs(img)
                        result.append(output)
            result = torch.stack(result, dim=0)
            save_image(result, "grid.png")

    def generate(
        self,
        total: int,
        batchsize: int,
        folder: str = "generate",
        tfs: transforms.Compose = None,
    ):
        assert total % batchsize == 0
        print("Start generating")
        self.model.eval()
        os.makedirs(folder, exist_ok=True)
        with torch.no_grad():
            for batchIdx in tqdm(range(total // batchsize)):
                x = torch.randn(batchsize, 3, 32, 32).cuda()
                for i in range(self.n_steps):
                    t = torch.tensor(self.n_steps - i - 1.0, device="cuda").long()
                    pred_noise = self.model(x, t.unsqueeze(0))
                    x = self.p_xt(x, pred_noise, t.unsqueeze(0))
                for idx, image in enumerate(x):
                    path = "{}/{:05d}.png".format(
                        folder,
                        batchIdx * batchsize + idx + 1,
                    )
                    if tfs is not None:
                        image = tfs(image)
                    save_image(image, path)
