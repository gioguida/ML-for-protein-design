"""Linear CKA across model variants on DMS embeddings.

Centered Kernel Alignment is a geometric-similarity measure between two
representation spaces, invariant to orthogonal rotation and isotropic
scaling. A pairwise 4×4 heatmap on the DMS subset tells you quantitatively
how much each fine-tuning step changed the embedding geometry.

Pipeline
--------
1. Load per-variant ``.npz`` from ``extract_embeddings.py``.
2. Restrict to DMS rows; intersect across variants for valid (non-NaN)
   embeddings (positional alignment is guaranteed by extract_embeddings.py's
   identical sampling seed).
3. Pre-reduce each model's DMS embeddings to PCA-50 within that model
   (each model uses its own PCA basis — for noise reduction only; CKA is
   rotation-invariant so the per-model basis doesn't affect the result).
4. Compute pairwise linear CKA → symmetric matrix with diagonal 1.

Reads
-----
N .npz files written by ``extract_embeddings.py`` (one per variant).

Writes
------
- ``cka_{emb_type}.csv`` — pairwise matrix (rows/cols = variant labels).
- ``cka_{emb_type}.png`` — annotated heatmap.

Run for both ``cdrh3_embs`` and ``whole_seq_embs``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SEED = 42
EMB_TYPES = ["cdrh3", "whole_seq"]
PCA_DIM = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_cka")


def load_variants(npz_paths: list[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for path in npz_paths:
        z = np.load(path, allow_pickle=False)
        variant = str(z["model_variant"][0])
        if variant in out:
            raise ValueError(f"Duplicate model_variant '{variant}' in inputs")
        out[variant] = {k: z[k] for k in z.files}
        log.info("Loaded %s (%d rows) from %s", variant, len(out[variant]["sequences"]), path)
    return out


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two centred representation matrices.

    Uses the Frobenius-norm formulation, which avoids forming the n×n Gram
    matrices when d ≤ n (cheap when X, Y are PCA-50).

    CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    norm_x = np.linalg.norm(Xc.T @ Xc, ord="fro")
    norm_y = np.linalg.norm(Yc.T @ Yc, ord="fro")
    if norm_x == 0 or norm_y == 0:
        return float("nan")
    return float(cross / (norm_x * norm_y))


def reduce_to_pca50(emb: np.ndarray, n_components: int) -> np.ndarray:
    n_components = min(n_components, emb.shape[0], emb.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    return pca.fit_transform(emb).astype(np.float64)


def shared_dms_mask(
    data: Dict[str, Dict[str, np.ndarray]], emb_key: str
) -> np.ndarray:
    """Boolean mask over the DMS subset: True iff embedding is finite in all variants."""
    variants = list(data.keys())
    n_dms = int((data[variants[0]]["source_labels"] == "dms").sum())
    mask = np.ones(n_dms, dtype=bool)
    for v in variants:
        d = data[v]
        is_dms = d["source_labels"] == "dms"
        n = int(is_dms.sum())
        if n != n_dms:
            raise ValueError(
                f"DMS row count mismatch: {variants[0]}={n_dms} vs {v}={n}. "
                "Reference set must be sampled identically across variants."
            )
        valid = ~np.isnan(d[emb_key][is_dms]).any(axis=1)
        mask &= valid
    return mask


def compute_pairwise(
    data: Dict[str, Dict[str, np.ndarray]], emb_type: str
) -> Tuple[List[str], np.ndarray, int]:
    emb_key = f"{emb_type}_embs"
    variants = list(data.keys())
    mask = shared_dms_mask(data, emb_key)
    n = int(mask.sum())
    log.info("[%s] shared valid DMS rows: %d / %d", emb_type, n, len(mask))

    reduced: Dict[str, np.ndarray] = {}
    for v in variants:
        d = data[v]
        is_dms = d["source_labels"] == "dms"
        emb = d[emb_key][is_dms][mask]
        reduced[v] = reduce_to_pca50(emb, PCA_DIM)
        log.info("[%s] %s: reduced to %d dims", emb_type, v, reduced[v].shape[1])

    k = len(variants)
    cka = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            val = linear_cka(reduced[variants[i]], reduced[variants[j]])
            cka[i, j] = val
            cka[j, i] = val
    return variants, cka, n


def save_heatmap(
    variants: List[str], cka: np.ndarray, n_rows: int, emb_type: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(0.9 * len(variants) + 3.5, 0.9 * len(variants) + 2.5))
    im = ax.imshow(cka, vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Linear CKA")
    ax.set_xticks(range(len(variants)))
    ax.set_yticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_yticklabels(variants)
    for i in range(len(variants)):
        for j in range(len(variants)):
            txt_color = "white" if cka[i, j] < 0.5 else "black"
            ax.text(j, i, f"{cka[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color=txt_color)
    ax.set_title(
        f"Linear CKA — {emb_type}  (n={n_rows} DMS rows, PCA-{PCA_DIM} per model)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "npz_files",
        nargs="+",
        type=Path,
        help="Per-variant .npz files from extract_embeddings.py",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    data = load_variants(args.npz_files)
    for emb_type in EMB_TYPES:
        variants, cka, n = compute_pairwise(data, emb_type)
        df = pd.DataFrame(cka, index=variants, columns=variants)
        csv_path = args.output_dir / f"cka_{emb_type}.csv"
        df.to_csv(csv_path)
        log.info("Wrote %s\n%s", csv_path, df.round(3).to_string())
        save_heatmap(variants, cka, n, emb_type,
                     args.output_dir / f"cka_{emb_type}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
