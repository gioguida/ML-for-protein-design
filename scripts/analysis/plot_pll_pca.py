"""Render figures for the joint PLL PCA analysis.

Reads
-----
- ``pll_pca.npz`` from ``compute_pll_pca.py``.

Writes (200 DPI, in ``{output_dir}/``)
-------------------------------------
- ``pll_c1_biplot_{fitness}.png`` — sequence-level joint PCA. Sequences as
  points colored by enrichment; model variants drawn as loading arrows.
- ``pll_c2_scores_{fitness}.png`` — per-position joint PCA scores in PC1/PC2,
  colored by enrichment. Quick view of which sequences load on the
  position-by-model axes.
- ``pll_c2_loadings.png`` — heatmap of C2 loadings. One row per principal
  component (top 4), columns are 24 CDR-H3 positions, color-coded by model.
  This is the scientifically richest view: the magnitude of each loading
  tells you which (model, position) pairs drive each PC.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_pll_pca")

FITNESS = [
    ("dms_M22_enrich", "M22 binding enrichment", "M22"),
    ("dms_SI06_enrich", "SI06 binding enrichment", "SI06"),
]


def biplot_c1(z: dict, fitness_key: str, fitness_label: str, out_path: Path) -> None:
    coords = z["c1_coords"]
    loadings = z["c1_loadings"]  # (n_components, M)
    variants = [str(v) for v in z["model_variants"]]
    ev = z["c1_explained_variance"]
    fitness = z[fitness_key]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    valid = ~np.isnan(fitness)
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1], s=10, c="lightgrey",
                   alpha=0.4, zorder=2)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], c=fitness[valid],
                    cmap="viridis", s=14, alpha=0.85, zorder=3)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=fitness_label)

    # Model loading arrows. Scale to ~80% of coord range so they are visible.
    lim = max(abs(coords).max(), 1e-6)
    arrow_scale = 0.7 * lim / max(abs(loadings).max(), 1e-6)
    for m_idx, label in enumerate(variants):
        dx = loadings[0, m_idx] * arrow_scale
        dy = loadings[1, m_idx] * arrow_scale if loadings.shape[0] > 1 else 0.0
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.6),
        )
        ax.text(dx * 1.08, dy * 1.08, label, color="red",
                ha="center", va="center", fontsize=10, fontweight="bold")

    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=1)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=1)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% var)" if len(ev) > 1 else "PC2")
    ax.set_title(f"PLL biplot (C1) — colored by {fitness_label}\n"
                 f"sequences = points; models = arrows", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def scores_c2(z: dict, fitness_key: str, fitness_label: str, out_path: Path) -> None:
    coords = z["c2_coords"]
    ev = z["c2_explained_variance"]
    fitness = z[fitness_key]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    valid = ~np.isnan(fitness)
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1], s=10, c="lightgrey",
                   alpha=0.4, zorder=2)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], c=fitness[valid],
                    cmap="viridis", s=14, alpha=0.85, zorder=3)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=fitness_label)

    # Annotate Spearman ρ of PC1 against fitness in the corner.
    x = coords[valid, 0]
    y = fitness[valid]
    if len(x) > 1:
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        ax.text(
            0.02, 0.97,
            f"PC1 vs {fitness_label}\n"
            f"Pearson r = {r_p:.3f} (p = {p_p:.1e})\n"
            f"Spearman ρ = {r_s:.3f} (p = {p_s:.1e})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
        )

    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=1)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=1)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% var)" if len(ev) > 1 else "PC2")
    ax.set_title(f"PLL per-position joint PCA (C2) — scores, colored by {fitness_label}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def loadings_heatmap(z: dict, out_path: Path) -> None:
    loadings = z["c2_loadings"]  # (n_components, M, P)
    variants = [str(v) for v in z["model_variants"]]
    ev = z["c2_explained_variance"]
    n_components, M, P = loadings.shape

    vmax = float(np.abs(loadings).max())
    fig, axes = plt.subplots(n_components, 1, figsize=(0.45 * P + 3.0, 1.4 * M * n_components + 1.0),
                             squeeze=False)
    for c in range(n_components):
        ax = axes[c, 0]
        im = ax.imshow(loadings[c], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(M))
        ax.set_yticklabels(variants)
        ax.set_xticks(range(P))
        ax.set_xticklabels([f"{i + 1}" for i in range(P)])
        ax.set_xlabel("CDR-H3 position (1-indexed)")
        ax.set_title(f"PC{c + 1}  ({100 * ev[c]:.1f}% var)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.020, pad=0.02, label="loading")
    fig.suptitle(
        "C2 loadings — which (model, CDR position) pairs drive each PC?",
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
    p.add_argument("--input", type=Path, required=True,
                   help="pll_pca.npz from compute_pll_pca.py")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    z = dict(np.load(args.input, allow_pickle=False))
    for fkey, flabel, fshort in FITNESS:
        biplot_c1(z, fkey, flabel,
                  args.output_dir / f"pll_c1_biplot_{fshort}.png")
        scores_c2(z, fkey, flabel,
                  args.output_dir / f"pll_c2_scores_{fshort}.png")
    loadings_heatmap(z, args.output_dir / "pll_c2_loadings.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
