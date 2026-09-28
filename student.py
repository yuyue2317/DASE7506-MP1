"""Student model E02: GPT with SwiGLU and Rotary Position Embeddings (RoPE)."""

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()

        hidden = int(8 * width / 3)
        hidden = ((hidden + 15) // 16) * 16

        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.out(
            F.silu(self.gate(x)) * self.value(x)
        )


def apply_rope(x):
    """
    Apply rotary position embeddings.

    x shape:
        [batch, heads, length, head_dim]
    """
    _, _, length, head_dim = x.shape

    # RoPE rotates pairs of dimensions, so head_dim must be even.
    assert head_dim % 2 == 0

    positions = torch.arange(
        length,
        device=x.device,
        dtype=torch.float32
    )

    inv_freq = 1.0 / (
        10000.0 ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=x.device,
                dtype=torch.float32
            ) / head_dim
        )
    )

    angles = positions[:, None] * inv_freq[None, :]

    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x_float = x.float()

    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    rotated = torch.stack(
        (rotated_even, rotated_odd),
        dim=-1
    ).flatten(-2)

    return rotated.to(dtype=x.dtype)


class Block(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()

        self.heads = heads

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

        self.mlp = SwiGLU(width)

    def forward(self, x):
        batch, length, width = x.shape

        q, k, v = (
            self.qkv(self.norm1(x))
            .view(
                batch,
                length,
                3,
                self.heads,
                width // self.heads
            )
            .permute(2, 0, 3, 1, 4)
        )

        # RoPE is applied to queries and keys, not values.
        q = apply_rope(q)
        k = apply_rope(k)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        x = x + self.proj(
            attended.transpose(1, 2)
            .reshape(batch, length, width)
        )

        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = dict(config)
        self.context = config["context"]

        width = config["width"]

        self.token = nn.Embedding(
            config["vocab"],
            width
        )

        # E02 difference:
        # No learned absolute position embedding.
        # Position information is injected through RoPE in attention.

        self.blocks = nn.ModuleList([
            Block(width, config["heads"])
            for _ in range(config["depth"])
        ])

        self.norm = nn.LayerNorm(width)

        self.head = nn.Linear(
            width,
            config["vocab"],
            bias=False
        )

        self.apply(self.initialize)

        # Weight tying.
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                std=0.02
            )

            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        # Unlike E01, no learned position embedding is added here.
        x = self.token(ids)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(
            self(ids).float(),
            dim=-1
        )


def build_model(config):
    return GPT(config)