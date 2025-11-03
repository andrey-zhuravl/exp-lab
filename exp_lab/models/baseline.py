from __future__ import annotations

def train(cfg, processed_dir: str, model_dir: str) -> None:
    from exp_lab.pipeline import _ttlab_train

    _ttlab_train(processed_dir, model_dir, cfg)
