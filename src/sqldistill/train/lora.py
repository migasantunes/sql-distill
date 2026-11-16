"""QLoRA fine-tuning of the student (M4).

``torch``, ``transformers``, ``peft`` and ``bitsandbytes`` will be imported
inside :func:`train` only, so this module imports with the core install.
"""

from __future__ import annotations

from pathlib import Path

from sqldistill.config import TrainConfig


def train(cfg: TrainConfig) -> Path:
    """Fine-tune the student with QLoRA and return the adapter directory.

    Intended steps (M4):

    1. Read the training JSONL and the eval benchmarks' db_ids and call
       ``splits.assert_disjoint`` -- a shared database is a hard failure.
    2. Format each example with ``train.format.to_chat_example`` and apply the
       tokenizer's chat template; drop (and count) examples over
       ``cfg.max_seq_len`` tokens.
    3. Load the base model in 4-bit (bitsandbytes NF4, ``cfg.load_in_4bit``);
       its weights stay frozen.
    4. Attach LoRA adapters (``cfg.lora_r``, ``cfg.lora_alpha``,
       ``cfg.lora_dropout``) with PEFT; only the adapters are trained.
    5. Mask the loss to the assistant turn: prompt tokens get label -100, so
       the model learns to write SQL, not to reproduce schemas.
    6. Train with ``cfg.lr``, ``cfg.epochs``, ``cfg.batch_size`` and
       ``cfg.grad_accum``, seeded by ``cfg.seed``; checkpoint every N steps so a
       dying free GPU session can resume; save the loss curve.
    7. Save the adapter and return its directory.
    """
    raise NotImplementedError("M4: QLoRA training loop (PEFT + bitsandbytes)")
