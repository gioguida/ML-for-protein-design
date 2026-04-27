"""Gibbs sampling diagnostics across model variants.

Three figures, plus a CSV summary:

1. **PLL distribution** — under each variant, compute CDR-only sequence PLL
   for that variant's Gibbs samples and for the DMS reference set. Side-by-
   side violins per variant. Answers: are Gibbs samples "as plausible" as
   real binders to the model, or are they pulled into low-PLL regions?

2. **Edit-distance from WT** — histogram of ``n_mutations`` per variant
   (already in the Gibbs CSV; no model needed). Shows how far each model
   walks from WT during sampling.

3. **Per-position mutation frequency** — heatmap (variants × 24 CDR
   positions) of the fraction of Gibbs snapshots whose residue at that
   position differs from C05 WT. Reveals whether sampling concentrates on
   particular positions or explores all of them.

Inputs
------
``--gibbs LABEL=CHECKPOINT=GIBBS_CSV`` repeated once per variant. CHECKPOINT
matches the same set of forms as ``compute_pll_pca.py``.

Writes
------
- ``gibbs_pll_dist.png``
- ``gibbs_edit_distance.png``
- ``gibbs_position_mutation_freq.png``
- ``gibbs_summary.csv``
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from protein_design.constants import (
    C05_CDRH3,
    C05_CDRH3_END,
    C05_CDRH3_START,
    add_context,
)

ESM2_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
SEED = 42
DEFAULT_DMS_M22 = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/D2_M22.csv"
DEFAULT_DMS_SI06 = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/D2_SI06.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gibbs_diagnostics")


# --------------------------------------------------------------------- model load
# Same prefix-stripping logic as compute_pll_pca.py — kept inline here so the
# script is standalone.


def _extract_state_dict(raw) -> dict:
    """Handle evotuning ('model_state_dict' wrapper) and DPO ('policy_state_dict'
    wrapper) checkpoint shapes alongside bare state dicts."""
    if isinstance(raw, dict):
        for key in ("policy_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def _load_pt_into_mlm(pt_path: Path) -> EsmForMaskedLM:
    log.info("Loading state-dict checkpoint %s", pt_path)
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(raw)
    new_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        new_state[k[len("model."):] if k.startswith("model.") else k] = v
    model = EsmForMaskedLM.from_pretrained(ESM2_MODEL_ID)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    non_optional_missing = [m for m in missing
                            if not m.startswith("esm.contact_head.")
                            and "position_ids" not in m]
    if non_optional_missing:
        raise RuntimeError(f"Checkpoint missing required keys: {non_optional_missing[:5]}")
    if unexpected:
        log.warning("Ignored %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    return model


def load_esm_for_mlm(checkpoint: str) -> EsmForMaskedLM:
    if not checkpoint:
        return EsmForMaskedLM.from_pretrained(ESM2_MODEL_ID)
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        return _load_pt_into_mlm(p)
    if p.is_dir():
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            return EsmForMaskedLM.from_pretrained(str(p))
        pt_path = next((p / n for n in ("best.pt", "final.pt") if (p / n).exists()), None)
        if pt_path is not None:
            return _load_pt_into_mlm(pt_path)
        raise FileNotFoundError(f"No HF weights, best.pt, or final.pt found at {checkpoint}")
    return EsmForMaskedLM.from_pretrained(checkpoint)


# --------------------------------------------------------------------- inference


@torch.no_grad()
def per_position_cdr_log_probs(
    model: EsmForMaskedLM,
    tokenizer,
    cdrh3_strings: List[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return (N, 24) log P(true aa | masked context) over each CDR-H3 position."""
    cdr_token_positions = list(range(C05_CDRH3_START + 1, C05_CDRH3_END + 1))
    P = len(cdr_token_positions)
    n = len(cdrh3_strings)
    out = np.zeros((n, P), dtype=np.float32)
    mask_id = tokenizer.mask_token_id

    full_vhs = [add_context(s) for s in cdrh3_strings]
    n_batches = (n + batch_size - 1) // batch_size
    for start in tqdm(range(0, n, batch_size), total=n_batches, desc="PLL"):
        batch = full_vhs[start:start + batch_size]
        spaced = [" ".join(list(s)) for s in batch]
        enc = tokenizer(spaced, return_tensors="pt", padding=True, add_special_tokens=True)
        tokens = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        B = tokens.shape[0]

        for p_idx, token_pos in enumerate(cdr_token_positions):
            masked = tokens.clone()
            masked[:, token_pos] = mask_id
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(input_ids=masked, attention_mask=attn).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            true_ids = tokens[:, token_pos]
            picks = log_probs[torch.arange(B, device=device), token_pos, true_ids]
            out[start:start + B, p_idx] = picks.cpu().numpy().astype(np.float32)
    return out


def sequence_pll(per_pos_log_probs: np.ndarray) -> np.ndarray:
    """Sum per-position log-probs to a sequence-level CDR PLL."""
    return per_pos_log_probs.sum(axis=1)


# ----------------------------------------------------------------- I/O helpers


def load_dms(m22_path: Path, si06_path: Path, max_n: int) -> List[str]:
    m22 = pd.read_csv(m22_path)[["aa", "M22_binding_enrichment_adj"]]
    si06 = pd.read_csv(si06_path)[["aa", "SI06_binding_enrichment_adj"]]
    merged = m22.merge(si06, on="aa", how="outer")
    if len(merged) > max_n:
        merged = merged.sample(n=max_n, random_state=SEED).reset_index(drop=True)
    return merged["aa"].astype(str).tolist()


def load_gibbs_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"chain_id", "gibbs_step", "cdrh3", "n_mutations"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gibbs CSV missing columns {missing}: {path}")
    df = df[df["cdrh3"].astype(str).str.len() == len(C05_CDRH3)].copy()
    return df


# ------------------------------------------------------------------------ plots


def plot_pll_violin(
    per_variant: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    variants = list(per_variant.keys())
    fig, ax = plt.subplots(figsize=(0.9 * len(variants) * 2 + 3.0, 4.6))

    positions = []
    data = []
    labels = []
    colors = []
    for i, v in enumerate(variants):
        gibbs = per_variant[v]["gibbs_pll"]
        dms = per_variant[v]["dms_pll"]
        positions.extend([2 * i + 1, 2 * i + 2])
        data.extend([dms, gibbs])
        labels.extend([f"{v}\nDMS", f"{v}\nGibbs"])
        colors.extend(["tab:blue", "tab:green"])

    parts = ax.violinplot(data, positions=positions, widths=0.7, showmeans=False,
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CDR-H3 sequence PLL")
    ax.set_title("Gibbs vs DMS PLL distribution (each variant scores under its own model)",
                 fontsize=11)
    ax.axhline(0, color="lightgrey", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_edit_distance(
    per_variant: Dict[str, Dict[str, np.ndarray]], out_path: Path
) -> None:
    variants = list(per_variant.keys())
    n_cols = len(variants)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.4 * n_cols, 4.0), squeeze=False)
    for col, v in enumerate(variants):
        nm = per_variant[v]["n_mutations"]
        ax = axes[0, col]
        ax.hist(nm, bins=range(0, int(nm.max()) + 2), color="tab:green",
                edgecolor="white", linewidth=0.4, align="left")
        ax.set_title(v, fontsize=11)
        ax.set_xlabel("edit distance from C05 WT")
        ax.set_ylabel("count")
        ax.set_xlim(left=-0.5)
    fig.suptitle("Gibbs samples — edit distance from C05 WT CDR-H3", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_position_mutation_freq(
    per_variant: Dict[str, Dict[str, np.ndarray]], out_path: Path
) -> None:
    variants = list(per_variant.keys())
    P = len(C05_CDRH3)
    matrix = np.zeros((len(variants), P), dtype=np.float32)
    for i, v in enumerate(variants):
        matrix[i] = per_variant[v]["pos_mut_freq"]

    fig, ax = plt.subplots(figsize=(0.45 * P + 3.5, 0.9 * len(variants) + 1.8))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(non-WT residue)")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xticks(range(P))
    ax.set_xticklabels([f"{i + 1}\n{a}" for i, a in enumerate(C05_CDRH3)], fontsize=8)
    ax.set_xlabel("CDR-H3 position (1-indexed; WT residue beneath)")
    ax.set_title("Per-position mutation frequency in Gibbs samples", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ------------------------------------------------------------------------ main


def parse_variant_spec(spec: str) -> Tuple[str, str, str]:
    parts = spec.split("=")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--gibbs must be of the form LABEL=CHECKPOINT=GIBBS_CSV, got: {spec!r}"
        )
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--gibbs",
        action="append",
        required=True,
        type=parse_variant_spec,
        help="LABEL=CHECKPOINT=GIBBS_CSV; repeat once per variant.",
    )
    p.add_argument("--dms-m22", default=DEFAULT_DMS_M22)
    p.add_argument("--dms-si06", default=DEFAULT_DMS_SI06)
    p.add_argument("--max-dms", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading DMS reference (max %d) …", args.max_dms)
    dms_cdrh3 = load_dms(Path(args.dms_m22), Path(args.dms_si06), args.max_dms)
    log.info("DMS: %d sequences", len(dms_cdrh3))

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    per_variant: Dict[str, Dict[str, np.ndarray]] = {}
    summary_rows: List[dict] = []

    for label, checkpoint, gibbs_csv in args.gibbs:
        log.info("=== %s ===", label)
        gdf = load_gibbs_csv(Path(gibbs_csv))
        gibbs_cdrh3 = gdf["cdrh3"].astype(str).tolist()
        n_mutations = gdf["n_mutations"].to_numpy(dtype=np.int32)
        log.info("Gibbs samples: %d (median edit distance from WT = %d)",
                 len(gibbs_cdrh3), int(np.median(n_mutations)))

        model = load_esm_for_mlm(checkpoint).eval().to(device)
        for p_ in model.parameters():
            p_.requires_grad = False
        if device.type == "cuda":
            model = model.half()

        gibbs_per_pos = per_position_cdr_log_probs(
            model, tokenizer, gibbs_cdrh3, device, args.batch_size,
        )
        gibbs_pll = sequence_pll(gibbs_per_pos)

        dms_per_pos = per_position_cdr_log_probs(
            model, tokenizer, dms_cdrh3, device, args.batch_size,
        )
        dms_pll = sequence_pll(dms_per_pos)

        # Per-position mutation frequency (vs C05 WT, no model needed).
        wt = np.array(list(C05_CDRH3))
        cdrh3_chars = np.array([list(s) for s in gibbs_cdrh3])  # (N, 24)
        pos_mut_freq = (cdrh3_chars != wt[None, :]).mean(axis=0).astype(np.float32)

        per_variant[label] = {
            "gibbs_pll": gibbs_pll,
            "dms_pll": dms_pll,
            "n_mutations": n_mutations,
            "pos_mut_freq": pos_mut_freq,
        }

        # Per-position residue counts for the summary CSV (top off-WT residue per pos).
        top_alts = []
        for p_idx in range(len(C05_CDRH3)):
            alt_counts = Counter(c for c in cdrh3_chars[:, p_idx] if c != C05_CDRH3[p_idx])
            top = alt_counts.most_common(1)
            top_alts.append(f"{top[0][0]}({top[0][1]})" if top else "-")

        summary_rows.append({
            "variant": label,
            "n_gibbs_samples": int(len(gibbs_cdrh3)),
            "edit_dist_median": float(np.median(n_mutations)),
            "edit_dist_max": int(n_mutations.max()),
            "gibbs_pll_median": float(np.median(gibbs_pll)),
            "dms_pll_median": float(np.median(dms_pll)),
            "delta_pll_gibbs_minus_dms": float(np.median(gibbs_pll) - np.median(dms_pll)),
            "max_pos_mut_freq": float(pos_mut_freq.max()),
            "top_alt_per_position": " ".join(top_alts),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_pll_violin(per_variant, args.output_dir / "gibbs_pll_dist.png")
    plot_edit_distance(per_variant, args.output_dir / "gibbs_edit_distance.png")
    plot_position_mutation_freq(per_variant, args.output_dir / "gibbs_position_mutation_freq.png")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "gibbs_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Wrote %s\n%s", summary_path,
             summary_df.drop(columns=["top_alt_per_position"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
