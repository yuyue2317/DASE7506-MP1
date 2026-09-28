"""Student model E01: GPT with a parameter-matched SwiGLU feed-forward network."""

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()

        # Parameter-match the baseline 4x GELU FFN approximately.
        # Baseline FFN:  d -> 4d -> d      ~= 8d^2 parameters
        # SwiGLU FFN:    two d -> h paths
        #                plus h -> d       ~= 3dh parameters
        # Therefore h ~= 8d/3.
        hidden = int(8 * width / 3)

        # Round up to a multiple of 16.
        hidden = ((hidden + 15) // 16) * 16

        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.out(
            F.silu(self.gate(x)) * self.value(x)
        )


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

        self.pos = nn.Embedding(
            self.context,
            width
        )

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

        # Share input embedding and output prediction weights.
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
        positions = torch.arange(
            ids.shape[1],
            device=ids.device
        )

        x = self.token(ids) + self.pos(positions)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def forward(self, ids):
        """Return unnormalized next-token logits."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Return normalized log probabilities for evaluation."""
        return F.log_softmax(
            self(ids).float(),
            dim=-1
        )


def build_model(config):
    return GPT(config)