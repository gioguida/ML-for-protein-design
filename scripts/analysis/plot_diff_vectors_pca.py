"""Render diff-vector PCA figures.

Reads
-----
- ``diff_pca_{cdrh3,whole_seq}.npz`` from ``compute_diff_vectors_pca.py``.

Writes (200 DPI, in ``{output_dir}/``)
-------------------------------------
- ``diff_pca_pc1_pc2_{emb_type}_{fitness}.png`` — one panel per variant,
  PC1 vs PC2 of diff-vectors colored by fitness (M22 or SI06).
- ``diff_norms_{emb_type}.png`` — one panel per variant, histogram of
  diff-vector Euclidean norms. **Diagnostic plot:** if norms are clustered
  near zero, the diff PCA is dominated by noise and the colored scatter
  should not be over-interpreted.

By construction, each model has its own WT embedding so diff-vectors live in
**different** coordinate systems across panels. Distances and axes are NOT
comparable across panels.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

EMB_TYPES = ["cdrh3", "whole_seq"]
FITNESS = [
    ("M22_enrich", "M22 binding enrichment", "M22"),
    ("SI06_enrich", "SI06 binding enrichment", "SI06"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_diff_vectors_pca")


def load(npz_path: Path) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
    z = np.load(npz_path, allow_pickle=False)
    variants = [str(v) for v in z["model_variants"]]
    per_variant: Dict[str, Dict[str, np.ndarray]] = {}
    for v in variants:
        per_variant[v] = {
            "diff_pca": z[f"{v}__diff_pca"],
            "diff_norms": z[f"{v}__diff_norms"],
            "M22_enrich": z[f"{v}__M22_enrich"],
            "SI06_enrich": z[f"{v}__SI06_enrich"],
            "cdrh3_identity": z[f"{v}__cdrh3_identity"],
            "explained_variance": z[f"{v}__pca_explained_variance"],
        }
    return variants, per_variant


def _scatter_continuous(ax, coords: np.ndarray, values: np.ndarray):
    valid_xy = ~np.isnan(coords).any(axis=1)
    valid_v = ~np.isnan(values)
    grey_mask = valid_xy & ~valid_v
    color_mask = valid_xy & valid_v
    if grey_mask.any():
        ax.scatter(coords[grey_mask, 0], coords[grey_mask, 1],
                   s=8, c="lightgrey", alpha=0.35, zorder=2)
    if color_mask.any():
        return ax.scatter(coords[color_mask, 0], coords[color_mask, 1],
                          c=values[color_mask], cmap="viridis",
                          s=14, alpha=0.85, zorder=3)
    return None


def make_pc1_pc2_grid(per_variant, variants, fitness_key, fitness_label, out_path):
    n_cols = len(variants)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.4), squeeze=False)
    for col, v in enumerate(variants):
        d = per_variant[v]
        ax = axes[0, col]
        ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=1)
        ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=1)
        sc = _scatter_continuous(ax, d["diff_pca"][:, :2], d[fitness_key])
        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=fitness_label)
        ev = d["explained_variance"]
        ax.set_title(v, fontsize=11)
        ax.set_xlabel(f"diff-PC1 ({100 * ev[0]:.1f}% var)")
        ax.set_ylabel(f"diff-PC2 ({100 * ev[1]:.1f}% var)" if len(ev) > 1 else "diff-PC2")
    fig.suptitle(
        f"Diff-vector PCA — DMS embeddings minus WT embedding, colored by {fitness_label}\n"
        f"(per-model coordinate systems; axes NOT comparable across panels)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_norm_histograms(per_variant, variants, out_path):
    n_cols = len(variants)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.0), squeeze=False)
    for col, v in enumerate(variants):
        d = per_variant[v]
        norms = d["diff_norms"]
        norms = norms[~np.isnan(norms)]
        ax = axes[0, col]
        ax.hist(norms, bins=40, color="tab:blue", edgecolor="white", linewidth=0.4)
        med = float(np.median(norms))
        ax.axvline(med, color="red", linestyle="--", linewidth=1.2,
                   label=f"median = {med:.3f}")
        ax.set_title(v, fontsize=11)
        ax.set_xlabel("‖embed(variant) − embed(WT)‖")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        "Diff-vector norms — diagnostic for whether diff PCA carries signal",
        fontsize=12,
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
        "--projections-dir",
        type=Path,
        required=True,
        help="Directory containing diff_pca_{emb_type}.npz",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for emb_type in EMB_TYPES:
        npz_path = args.projections_dir / f"diff_pca_{emb_type}.npz"
        if not npz_path.exists():
            log.warning("Skipping %s — %s not found", emb_type, npz_path)
            continue
        variants, per_variant = load(npz_path)
        for fkey, flabel, fshort in FITNESS:
            make_pc1_pc2_grid(
                per_variant, variants, fkey, flabel,
                args.output_dir / f"diff_pca_pc1_pc2_{emb_type}_{fshort}.png",
            )
        make_norm_histograms(
            per_variant, variants,
            args.output_dir / f"diff_norms_{emb_type}.png",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
