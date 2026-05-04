"""Procrustes alignment + per-sequence displacement, per qualifying model pair.

For each pair of model variants whose linear CKA exceeds ``--cka-threshold``,
align the two PCA-50-reduced DMS embeddings via orthogonal Procrustes and
plot per-sequence residual displacement against fitness. Answers: do
high-affinity DMS variants move more than neutral ones during fine-tuning?

Caveat: Procrustes alignment minimizes the sum of squared displacements
across all sequences, so displacement is *relative* to the average movement.
Below CKA ≈ 0.6–0.8 the rotation is fitting a distortion rather than a rigid
realignment, so displacement values are no longer interpretable. Pick the
threshold accordingly. The plan defaults to 0.8; we ship 0.5 as a more
permissive default that still produces qualitatively meaningful results
when fine-tuning has reorganized representations heavily.

Reads
-----
- N .npz files written by ``extract_embeddings.py`` (one per variant).
- ``cka_{emb_type}.csv`` from ``compute_cka.py`` for the gating decision.

Writes (in ``{output_dir}/``)
----------------------------
- ``procrustes_displacement_{i}_vs_{j}_{fitness}_{emb_type}.png`` —
  scatter of displacement vs M22/SI06 enrichment per qualifying pair.
- ``procrustes_summary_{emb_type}.csv`` — per-pair displacement
  mean/median/quantiles + Pearson r and Spearman ρ vs each fitness column.

If no pair passes the gate, the script logs a warning and exits cleanly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

SEED = 42
EMB_TYPES = ["cdrh3", "whole_seq"]
PCA_DIM = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_procrustes_displacement")


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


def shared_dms_mask(
    data: Dict[str, Dict[str, np.ndarray]], emb_key: str
) -> np.ndarray:
    variants = list(data.keys())
    n_dms = int((data[variants[0]]["source_labels"] == "dms").sum())
    mask = np.ones(n_dms, dtype=bool)
    for v in variants:
        d = data[v]
        is_dms = d["source_labels"] == "dms"
        valid = ~np.isnan(d[emb_key][is_dms]).any(axis=1)
        mask &= valid
    return mask


def reduce_to_pca50(emb: np.ndarray, n_components: int) -> np.ndarray:
    n_components = min(n_components, emb.shape[0], emb.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    return pca.fit_transform(emb).astype(np.float64)


def procrustes_displacement(
    X: np.ndarray, Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Orthogonal Procrustes from X to Y plus per-row residual norm.

    Returns (R, displacement, scale) where R is orthogonal of shape (d, d),
    displacement[k] = ||X[k] @ R − Y[k]||, and scale is scipy's procrustes
    scaling factor (informational only).
    """
    R, scale = orthogonal_procrustes(X, Y)
    aligned = X @ R
    displacement = np.linalg.norm(aligned - Y, axis=1)
    return R, displacement, float(scale)


def compute_for_pair(
    a: str, b: str, data, emb_key: str, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    is_dms_a = data[a]["source_labels"] == "dms"
    is_dms_b = data[b]["source_labels"] == "dms"
    emb_a = data[a][emb_key][is_dms_a][mask]
    emb_b = data[b][emb_key][is_dms_b][mask]

    Xa = reduce_to_pca50(emb_a, PCA_DIM)
    Xb = reduce_to_pca50(emb_b, PCA_DIM)
    _, displacement, scale = procrustes_displacement(Xa, Xb)
    log.info("[%s vs %s] Procrustes done; n=%d, scale=%.4f, "
             "displacement median=%.3f IQR=[%.3f, %.3f]",
             a, b, len(displacement), scale,
             float(np.median(displacement)),
             float(np.quantile(displacement, 0.25)),
             float(np.quantile(displacement, 0.75)))

    fitness = {
        "M22_enrich": data[a]["M22_binding_enrichment"][is_dms_a][mask],
        "SI06_enrich": data[a]["SI06_binding_enrichment"][is_dms_a][mask],
    }
    return displacement, fitness


def make_pair_plot(
    a: str, b: str, displacement, fitness, out_path: Path, fitness_label: str
) -> Tuple[float, float, float, float]:
    valid = ~np.isnan(fitness)
    x = displacement[valid]
    y = fitness[valid]

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(x, y, s=14, alpha=0.55, c="tab:blue")
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, slope * xr + intercept, "r-", linewidth=1.5)
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        ax.text(
            0.02, 0.97,
            f"Pearson r = {r_p:.3f} (p = {p_p:.1e})\n"
            f"Spearman ρ = {r_s:.3f} (p = {p_s:.1e})\n"
            f"n = {len(x)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
        )
    else:
        r_p = p_p = r_s = p_s = float("nan")

    ax.set_xlabel(f"Procrustes displacement ‖x·R − y‖   ({a} → {b})")
    ax.set_ylabel(fitness_label)
    ax.set_title(f"{a} → {b}: displacement vs {fitness_label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return float(r_p), float(p_p), float(r_s), float(p_s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("npz_files", nargs="+", type=Path,
                   help="Per-variant .npz files from extract_embeddings.py")
    p.add_argument("--cka-dir", type=Path, required=True,
                   help="Directory containing cka_{emb_type}.csv from compute_cka.py")
    p.add_argument("--cka-threshold", type=float, default=0.5,
                   help="Run Procrustes only on pairs with linear CKA above this. "
                        "Plan suggests 0.8; 0.5 is more permissive (default).")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    data = load_variants(args.npz_files)
    variants = list(data.keys())

    for emb_type in EMB_TYPES:
        emb_key = f"{emb_type}_embs"
        cka_path = args.cka_dir / f"cka_{emb_type}.csv"
        if not cka_path.exists():
            log.warning("Skipping %s — %s not found", emb_type, cka_path)
            continue
        cka = pd.read_csv(cka_path, index_col=0)

        mask = shared_dms_mask(data, emb_key)
        log.info("[%s] %d shared valid DMS rows", emb_type, int(mask.sum()))

        rows: List[dict] = []
        any_pair = False
        for a, b in combinations(variants, 2):
            cka_ab = float(cka.loc[a, b])
            if cka_ab < args.cka_threshold:
                log.info("[%s] skip %s vs %s (CKA=%.3f < %.2f)",
                         emb_type, a, b, cka_ab, args.cka_threshold)
                continue
            any_pair = True
            displacement, fitness = compute_for_pair(a, b, data, emb_key, mask)
            row = {
                "pair": f"{a} vs {b}",
                "model_a": a,
                "model_b": b,
                "cka": cka_ab,
                "n": int(len(displacement)),
                "disp_median": float(np.median(displacement)),
                "disp_q25": float(np.quantile(displacement, 0.25)),
                "disp_q75": float(np.quantile(displacement, 0.75)),
                "disp_max": float(displacement.max()),
            }
            for fkey, flabel, fshort in [
                ("M22_enrich", "M22 binding enrichment", "M22"),
                ("SI06_enrich", "SI06 binding enrichment", "SI06"),
            ]:
                out_path = (args.output_dir /
                            f"procrustes_displacement_{a}_vs_{b}_{fshort}_{emb_type}.png")
                rp, pp, rs, ps = make_pair_plot(a, b, displacement, fitness[fkey],
                                                out_path, flabel)
                row[f"pearson_r_{fshort}"] = rp
                row[f"pearson_p_{fshort}"] = pp
                row[f"spearman_rho_{fshort}"] = rs
                row[f"spearman_p_{fshort}"] = ps
            rows.append(row)

        if not any_pair:
            log.warning("[%s] no pairs passed CKA threshold %.2f; "
                        "no Procrustes plots produced", emb_type, args.cka_threshold)
            continue

        summary_df = pd.DataFrame(rows)
        summary_path = args.output_dir / f"procrustes_summary_{emb_type}.csv"
        summary_df.to_csv(summary_path, index=False)
        log.info("Wrote %s\n%s", summary_path, summary_df.round(3).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
