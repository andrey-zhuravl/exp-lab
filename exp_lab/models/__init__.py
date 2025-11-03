from __future__ import annotations

from .baseline import train as train_baseline
from .hyperbolic import train as train_hyperbolic
from .linear_attn import train as train_linear


REGISTRY = {
    "baseline": train_baseline,
    "linear_attention": train_linear,
    "hyperbolic": train_hyperbolic,
}

REG = REGISTRY


def get_trainer(arch: str):
    return REGISTRY.get(arch)
