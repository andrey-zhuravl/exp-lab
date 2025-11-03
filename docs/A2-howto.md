# A2 HOWTO

- Add a new recipe: reference it in sweep.grid (e.g., dataset.recipe) or set in base YAML.
- Add a new model: implement `exp_lab/models/<name>.py` and register in `REG`. Use `model.arch: <name>`.
- Add an ablation: extend `sweeps.py` dot-path override and handle it in `pipeline.py` before training.
- Reuse tokenizers: ensure spec keys (recipe, size, seed, vocab params) remain identical; registry will match by hash.
