import torch
import torch.nn as nn
import torch.nn.functional as F


def pad2d(x, pad, mode: str = "circular"):
    if mode in ("zero", "zeros", "constant", None):
        return F.pad(x, pad, mode="constant")
    return F.pad(x, pad, mode=mode)


def pad_to_multiple(x, multiple, mode: str = "circular"):
    """Pad (B,C,H,W) so H and W are multiples of ``multiple``."""
    B, C, H, W = x.shape
    Hp = ((H + multiple - 1) // multiple) * multiple
    Wp = ((W + multiple - 1) // multiple) * multiple
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h == 0 and pad_w == 0:
        return x, (H, W)
    x = pad2d(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (H, W)


def window_partition(x, win):
    B, C, H, W = x.shape
    x = x.view(B, C, H // win, win, W // win, win)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.view(-1, win * win, C)
    return x


def window_reverse(xw, win, H, W, C):
    Bnw, N, _ = xw.shape
    Nh, Nw = H // win, W // win
    B = Bnw // (Nh * Nw)
    xw = xw.view(B, Nh, Nw, win, win, C)
    xw = xw.permute(0, 5, 1, 3, 2, 4).contiguous()
    xw = xw.view(B, C, H, W)
    return xw


class ConvEmbed(nn.Module):
    """Patchless conv embed with configurable padding."""

    def __init__(self, in_ch, dim, k=3, pad_mode: str = "circular"):
        super().__init__()
        self.k = k
        self.pad_mode = pad_mode
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=k, padding=0, bias=False)

    def forward(self, x):
        p = self.k // 2
        x = pad2d(x, (p, p, p, p), mode=self.pad_mode)
        return self.proj(x)


class RelPosBias(nn.Module):
    def __init__(self, heads, win):
        super().__init__()
        self.heads = heads
        self.win = win
        N = win * win
        self.bias = nn.Parameter(torch.zeros(heads, (2 * win - 1) * (2 * win - 1)))
        coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing="ij"))
        coords = coords.view(2, -1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel[0] += win - 1
        rel[1] += win - 1
        index = rel[0] * (2 * win - 1) + rel[1]
        self.register_buffer("index", index.long(), persistent=False)

    def forward(self):
        N = self.win * self.win
        return self.bias[:, self.index].view(self.heads, N, N)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads=4, win=8, qk_scale=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.win = win
        self.scale = qk_scale or (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rpb = RelPosBias(heads, win)

    def forward(self, x):
        Bnw, N, C = x.shape
        qkv = self.qkv(x).view(Bnw, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3) * self.scale
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn + self.rpb()[None, :, :, :]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(Bnw, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, win=8, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, shifted=False, pad_mode: str = "circular"):
        super().__init__()
        self.win = win
        self.shifted = shifted
        self.pad_mode = pad_mode
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads=heads, win=win, dropout=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        orig_H, orig_W = H, W
        if self.shifted:
            shift = self.win // 2
            x = torch.roll(x, shifts=(-shift, -shift), dims=(-2, -1))
        x, _ = pad_to_multiple(x, self.win, mode=self.pad_mode)
        B, C, H, W = x.shape
        xw = window_partition(x, self.win)
        shortcut = xw
        xw = self.norm1(xw)
        xw = self.attn(xw)
        xw = shortcut + xw
        xw = xw + self.mlp(self.norm2(xw))
        x = window_reverse(xw, self.win, H, W, C)
        x = x[:, :, :orig_H, :orig_W]
        if self.shifted:
            shift = self.win // 2
            x = torch.roll(x, shifts=(shift, shift), dims=(-2, -1))
        return x


class BinarySwinPBC(nn.Module):
    """Swin-style backbone for masked token prediction."""

    def __init__(self, in_ch=2, dim=64, depth=4, heads=4, win=8, shifted=True, out_ch=1, pad_mode: str = "circular"):
        super().__init__()
        self.pad_mode = pad_mode
        self.embed = ConvEmbed(in_ch, dim, k=3, pad_mode=pad_mode)
        blocks = []
        for i in range(depth):
            blocks.append(SwinBlock(dim, heads=heads, win=win, shifted=(shifted and (i % 2 == 1)), pad_mode=pad_mode))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(dim, out_ch, kernel_size=1)

    def forward(self, x_in):
        h = self.embed(x_in)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(h)
        return logits
