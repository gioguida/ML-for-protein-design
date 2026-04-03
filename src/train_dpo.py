"""Hydra-driven DPO training entrypoint.

This script:
1) builds preference pairs from clustered D2 data
2) splits pairs into train/val/test
3) trains policy ESM2 with DPO against a frozen reference ESM2
4) logs metrics to console/file and optionally Weights & Biases
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .dataset import create_train_val_test_split, default_data_paths, load_dpo_pair_dataframe
    from .loss import dpo_loss
    from .model import ESM2PLLScorer
    from .utils import ModelConfig
else:  # pragma: no cover
    from dataset import create_train_val_test_split, default_data_paths, load_dpo_pair_dataframe
    from loss import dpo_loss
    from model import ESM2PLLScorer
    from utils import ModelConfig


def _load_hydra_modules() -> Tuple[Any, Any, Any, Any]:
    try:
        hydra = importlib.import_module("hydra")
        omegaconf = importlib.import_module("omegaconf")
        hydra_config_mod = importlib.import_module("hydra.core.hydra_config")
        hydra_utils_mod = importlib.import_module("hydra.utils")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Hydra/OmegaConf is required for train_dpo.py. Install hydra-core and omegaconf."
        ) from exc

    return hydra, omegaconf.OmegaConf, hydra_config_mod.HydraConfig, hydra_utils_mod.to_absolute_path


hydra, OmegaConf, HydraConfig, to_absolute_path = _load_hydra_modules()


class PairDataset(Dataset):
    """Minimal dataset of (chosen, rejected) sequence pairs."""

    def __init__(self, pairs_df: pd.DataFrame):
        self.pairs = list(zip(pairs_df["chosen_sequence"], pairs_df["rejected_sequence"]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


def _pair_collate(batch: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return list(batch)


def _setup_logger(output_dir: Path, level_name: str) -> logging.Logger:
    logger = logging.getLogger("dpo_train")
    logger.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _init_wandb(cfg: Any, output_dir: Path, logger: logging.Logger) -> Tuple[Optional[Any], Optional[Any]]:
    if not bool(cfg.wandb.enabled):
        return None, None

    try:
        wandb = importlib.import_module("wandb")
    except ModuleNotFoundError:  # pragma: no cover
        logger.warning("wandb is not available; continuing without Weights & Biases logging.")
        return None, None

    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
        name=None if cfg.wandb.run_name is None else str(cfg.wandb.run_name),
        tags=None if cfg.wandb.tags is None else list(cfg.wandb.tags),
        notes=None if cfg.wandb.notes is None else str(cfg.wandb.notes),
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return wandb, run


def _log_pair_diagnostics(logger: logging.Logger, pairs_df: pd.DataFrame, preview_count: int = 5) -> None:
    if pairs_df.empty:
        logger.warning("No pairs available after preprocessing.")
        return

    margins = pairs_df["delta_margin"].astype(float)
    logger.info(
        "Pair stats | n=%d | margin mean=%.4f median=%.4f min=%.4f max=%.4f",
        len(pairs_df),
        float(margins.mean()),
        float(margins.median()),
        float(margins.min()),
        float(margins.max()),
    )

    by_view = pairs_df.groupby("source_view").size().sort_values(ascending=False)
    logger.info("Pairs per view: %s", ", ".join(f"{k}:{int(v)}" for k, v in by_view.items()))

    show_n = min(int(preview_count), len(pairs_df))
    if show_n <= 0:
        return

    top_examples = pairs_df.nlargest(show_n, "delta_margin")
    low_examples = pairs_df.nsmallest(show_n, "delta_margin")

    logger.info("Top margin examples:")
    for _, row in top_examples.iterrows():
        logger.info(
            "  view=%s cluster=%s margin=%.4f chosen=%s rejected=%s",
            row["source_view"],
            row["cluster_idx"],
            float(row["delta_margin"]),
            row["chosen_sequence"],
            row["rejected_sequence"],
        )

    logger.info("Bottom margin examples:")
    for _, row in low_examples.iterrows():
        logger.info(
            "  view=%s cluster=%s margin=%.4f chosen=%s rejected=%s",
            row["source_view"],
            row["cluster_idx"],
            float(row["delta_margin"]),
            row["chosen_sequence"],
            row["rejected_sequence"],
        )


def _build_pairs_dataframe(cfg: Any) -> pd.DataFrame:
    defaults = default_data_paths()

    raw_csv_path = (
        defaults["raw_m22"]
        if cfg.data.raw_csv is None
        else Path(to_absolute_path(str(cfg.data.raw_csv)))
    )
    processed_dir = (
        defaults["processed_dir"]
        if cfg.data.processed_dir is None
        else Path(to_absolute_path(str(cfg.data.processed_dir)))
    )

    return load_dpo_pair_dataframe(
        pairing_strategy=str(cfg.data.pairing_strategy),
        include_views=[str(v) for v in cfg.data.include_views],
        raw_csv_path=raw_csv_path,
        processed_dir=processed_dir,
        force_rebuild=bool(cfg.data.force_rebuild),
        min_positive_delta=float(cfg.data.min_positive_delta),
        deduplicate_across_views=bool(cfg.data.deduplicate_across_views),
    )


def _build_dataloader(
    pairs_df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = PairDataset(pairs_df)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=_pair_collate,
        generator=generator,
        drop_last=False,
    )


def _build_scorers(cfg: Any) -> Tuple[ESM2PLLScorer, ESM2PLLScorer]:
    model_cfg = ModelConfig(
        esm_model_path=str(cfg.model.esm_model_path),
        device=str(cfg.training.device),
        use_context=bool(cfg.model.use_context),
    )

    policy = ESM2PLLScorer(model_cfg)
    reference = ESM2PLLScorer(model_cfg)

    for param in reference.model.parameters():
        param.requires_grad_(False)

    return policy, reference


def _build_optimizer_and_scheduler(
    cfg: Any,
    policy: ESM2PLLScorer,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable policy parameters found.")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    scheduler = None
    if bool(cfg.training.scheduler.enabled):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.training.scheduler.step_size),
            gamma=float(cfg.training.scheduler.gamma),
        )

    return optimizer, scheduler


def _save_checkpoint(
    path: Path,
    epoch: int,
    policy: ESM2PLLScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    best_val_loss: float,
) -> None:
    state = {
        "epoch": int(epoch),
        "policy_state_dict": policy.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "best_val_loss": float(best_val_loss),
    }
    torch.save(state, path)


def _load_checkpoint(
    checkpoint_path: Path,
    policy: ESM2PLLScorer,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> Tuple[int, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    policy.model.load_state_dict(ckpt["policy_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    next_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    return next_epoch, best_val_loss


def _run_epoch(
    policy: ESM2PLLScorer,
    reference: ESM2PLLScorer,
    dataloader: DataLoader,
    beta: float,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip_norm: float,
    logger: logging.Logger,
    log_every_n_steps: int,
) -> Dict[str, float]:
    is_train = optimizer is not None

    if is_train:
        policy.model.train()
    else:
        policy.model.eval()
    reference.model.eval()

    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0

    for step, batch in enumerate(dataloader, start=1):
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        try:
            loss = dpo_loss(
                batch,
                beta=beta,
                scorer=policy,
                reference=reference,
                policy_use_grad=is_train,
            )
        except ValueError:
            skipped_batches += 1
            continue

        if is_train:
            loss.backward()
            if grad_clip_norm > 0:
                clip_grad_norm_(policy.model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

        if is_train and log_every_n_steps > 0 and step % log_every_n_steps == 0:
            logger.info("step=%d train_loss=%.6f", step, float(loss.item()))

    avg_loss = total_loss / max(1, num_batches)
    return {
        "loss": float(avg_loss),
        "num_batches": float(num_batches),
        "skipped_batches": float(skipped_batches),
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Any) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(output_dir=output_dir, level_name=str(cfg.logging.level))
    logger.info("Starting DPO training run")

    resolved_cfg_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_cfg_path)
    logger.info("Saved resolved config to %s", resolved_cfg_path)

    wandb_mod, wandb_run = _init_wandb(cfg, output_dir, logger)

    pairs_df = _build_pairs_dataframe(cfg)
    if pairs_df.empty:
        raise ValueError("No DPO pairs generated. Adjust data pairing settings.")

    _log_pair_diagnostics(logger, pairs_df, preview_count=int(cfg.logging.preview_count))

    train_df, val_df, test_df = create_train_val_test_split(
        pairs_df,
        train_frac=float(cfg.data.train_frac),
        val_frac=float(cfg.data.val_frac),
        test_frac=float(cfg.data.test_frac),
        seed=int(cfg.seed),
    )
    logger.info("Split sizes | train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    train_loader = _build_dataloader(
        train_df,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        seed=int(cfg.seed),
        num_workers=int(cfg.training.num_workers),
    )
    val_loader = _build_dataloader(
        val_df,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        seed=int(cfg.seed) + 1,
        num_workers=int(cfg.training.num_workers),
    )
    test_loader = _build_dataloader(
        test_df,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        seed=int(cfg.seed) + 2,
        num_workers=int(cfg.training.num_workers),
    )

    policy, reference = _build_scorers(cfg)
    optimizer, scheduler = _build_optimizer_and_scheduler(cfg, policy)

    ckpt_dir = output_dir / str(cfg.checkpointing.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / str(cfg.checkpointing.best_filename)
    last_ckpt = ckpt_dir / str(cfg.checkpointing.last_filename)

    start_epoch = 1
    best_val_loss = float("inf")
    if cfg.training.resume_checkpoint is not None:
        resume_path = Path(to_absolute_path(str(cfg.training.resume_checkpoint)))
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_val_loss = _load_checkpoint(
            checkpoint_path=resume_path,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    history: List[Dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(start_epoch, int(cfg.training.num_epochs) + 1):
        train_metrics = _run_epoch(
            policy=policy,
            reference=reference,
            dataloader=train_loader,
            beta=float(cfg.training.beta),
            optimizer=optimizer,
            grad_clip_norm=float(cfg.training.grad_clip_norm),
            logger=logger,
            log_every_n_steps=int(cfg.logging.log_every_n_steps),
        )

        val_metrics = _run_epoch(
            policy=policy,
            reference=reference,
            dataloader=val_loader,
            beta=float(cfg.training.beta),
            optimizer=None,
            grad_clip_norm=0.0,
            logger=logger,
            log_every_n_steps=0,
        )

        if scheduler is not None:
            scheduler.step()

        lr = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": float(epoch),
            "lr": lr,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "train_batches": float(train_metrics["num_batches"]),
            "val_batches": float(val_metrics["num_batches"]),
            "train_skipped": float(train_metrics["skipped_batches"]),
            "val_skipped": float(val_metrics["skipped_batches"]),
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %d | lr=%.3e | train_loss=%.6f | val_loss=%.6f | train_batches=%d | val_batches=%d",
            epoch,
            lr,
            epoch_record["train_loss"],
            epoch_record["val_loss"],
            int(epoch_record["train_batches"]),
            int(epoch_record["val_batches"]),
        )

        if wandb_run is not None:
            wandb_mod.log(epoch_record, step=epoch)

        improved = epoch_record["val_loss"] < best_val_loss
        if improved:
            best_val_loss = epoch_record["val_loss"]
            epochs_without_improvement = 0
            _save_checkpoint(
                best_ckpt,
                epoch=epoch,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
            )
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            last_ckpt,
            epoch=epoch,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
        )

        if int(cfg.training.patience) > 0 and epochs_without_improvement >= int(cfg.training.patience):
            logger.info("Early stopping after %d epochs without val improvement.", epochs_without_improvement)
            break

    if bool(cfg.training.evaluate_best_checkpoint) and best_ckpt.exists():
        _load_checkpoint(best_ckpt, policy=policy, optimizer=None, scheduler=None)
        logger.info("Loaded best checkpoint for final test evaluation: %s", best_ckpt)

    test_metrics = _run_epoch(
        policy=policy,
        reference=reference,
        dataloader=test_loader,
        beta=float(cfg.training.beta),
        optimizer=None,
        grad_clip_norm=0.0,
        logger=logger,
        log_every_n_steps=0,
    )
    logger.info(
        "Test metrics | loss=%.6f | batches=%d | skipped=%d",
        float(test_metrics["loss"]),
        int(test_metrics["num_batches"]),
        int(test_metrics["skipped_batches"]),
    )

    history_df = pd.DataFrame(history)
    history_csv_path = output_dir / str(cfg.logging.history_csv)
    history_df.to_csv(history_csv_path, index=False)

    summary = {
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_metrics["loss"]),
        "test_batches": int(test_metrics["num_batches"]),
        "test_skipped_batches": int(test_metrics["skipped_batches"]),
        "num_train_pairs": int(len(train_df)),
        "num_val_pairs": int(len(val_df)),
        "num_test_pairs": int(len(test_df)),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved history to %s", history_csv_path)
    logger.info("Saved summary to %s", summary_path)

    if wandb_run is not None:
        wandb_mod.log({f"test/{k}": v for k, v in summary.items()})
        wandb_run.summary.update(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
