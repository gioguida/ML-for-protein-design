"""Diff-vector SVD — per variant, uncentered SVD on ``embed(variant) − embed(WT)``.

Why this exists
---------------
Mean-pooled CDR-H3 embeddings of ED2 sequences are dominated by the 22/24
positions that are identical to WT. Subtracting the WT embedding removes that
shared component and exposes only the change induced by the two mutations.

**Why SVD and not PCA.** ``sklearn.decomposition.PCA`` always re-centers the
input on its sample mean before fitting, which would undo the WT subtraction:
PCA on ``X`` and PCA on ``X − WT`` produce identical axes and would duplicate
the per-model DMS PCA (Phase 1). To preserve the "WT is at the origin"
interpretation, we use truncated SVD on the diff matrix without re-centering:
the dominant component then captures the *direction in which DMS sequences
collectively move away from WT*, not the dominant within-DMS variance axis.

Caveats (surfaced in plots downstream)
- Each model has its own WT embedding, so diff-vectors live in *different*
  coordinate systems across models. **No cross-model comparison is valid here.**
- For ED2 (only 2 mutations out of 24 CDR positions), diff norms may be tiny
  relative to embedding noise. The diff-norm histogram in the companion plot
  script is the diagnostic for this — if all norms are clustered near zero,
  the diff SVD is dominated by noise.

Reads
-----
N .npz files written by ``extract_embeddings.py`` (one per variant). The WT
row is identified via ``source_labels == "wt"`` (single row).

Writes
------
- ``diff_pca_{emb_type}.npz`` — per-variant 2D coords + diff norms +
  enrichment + per-variant explained variance, keyed ``{variant}__{field}``.
- ``diff_pca_{variant}_{emb_type}.pkl`` — fitted ``TruncatedSVD`` per variant.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

SEED = 42
EMB_TYPES = ["cdrh3", "whole_seq"]
N_COMPONENTS = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_diff_vectors_pca")


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


def fit_diff_svd(
    emb: np.ndarray, src: np.ndarray
) -> Tuple[TruncatedSVD, np.ndarray, np.ndarray]:
    """Subtract WT embedding from DMS embeddings, fit *uncentered* SVD on diffs.

    Returns
    -------
    svd : fitted TruncatedSVD (no re-centering — WT stays at the origin)
    coords : (n_dms, n_components) float32; NaN where DMS or WT embedding was missing
    norms : (n_dms,) float32; Euclidean norm of each diff-vector; NaN where missing
    """
    is_wt = src == "wt"
    if is_wt.sum() != 1:
        raise ValueError(f"Expected exactly one WT row, found {int(is_wt.sum())}")
    wt_emb = emb[is_wt][0]
    if np.isnan(wt_emb).any():
        raise ValueError("WT embedding contains NaN — cannot compute diff-vectors.")

    is_dms = src == "dms"
    n_dms = int(is_dms.sum())
    if n_dms == 0:
        raise ValueError("No DMS rows found in this variant's embeddings.")

    dms_emb = emb[is_dms]
    valid = ~np.isnan(dms_emb).any(axis=1)
    n_valid = int(valid.sum())
    if n_valid < 3:
        raise ValueError(f"Need at least 3 valid DMS rows; got {n_valid}")

    diffs = dms_emb[valid] - wt_emb[None, :]
    n_components = min(N_COMPONENTS, n_valid - 1, emb.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    svd.fit(diffs)

    coords = np.full((n_dms, n_components), np.nan, dtype=np.float32)
    coords[valid] = svd.transform(diffs).astype(np.float32)

    norms = np.full(n_dms, np.nan, dtype=np.float32)
    norms[valid] = np.linalg.norm(diffs, axis=1).astype(np.float32)

    return svd, coords, norms


def process_emb_type(
    emb_type: str,
    data: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
) -> None:
    emb_key = f"{emb_type}_embs"
    log.info("=== %s ===", emb_type)
    out: Dict[str, np.ndarray] = {}

    for variant, d in data.items():
        emb = d[emb_key]
        src = d["source_labels"]
        is_dms = src == "dms"

        svd, coords, norms = fit_diff_svd(emb, src)
        valid_norms = norms[~np.isnan(norms)]

        prefix = f"{variant}__"
        out[prefix + "diff_pca"] = coords
        out[prefix + "diff_norms"] = norms
        out[prefix + "M22_enrich"] = d["M22_binding_enrichment"][is_dms]
        out[prefix + "SI06_enrich"] = d["SI06_binding_enrichment"][is_dms]
        out[prefix + "cdrh3_identity"] = d["cdrh3_identity_to_wt"][is_dms]
        out[prefix + "sequences"] = d["sequences"][is_dms]
        out[prefix + "pca_explained_variance"] = svd.explained_variance_ratio_.astype(np.float32)

        ev = svd.explained_variance_ratio_
        log.info(
            "%s: fit on %d diffs; norm median=%.3f IQR=[%.3f, %.3f]; "
            "PC1=%.1f%% PC2=%.1f%% total(%d)=%.1f%%",
            variant,
            len(valid_norms),
            float(np.median(valid_norms)),
            float(np.quantile(valid_norms, 0.25)),
            float(np.quantile(valid_norms, 0.75)),
            100 * ev[0],
            100 * ev[1] if len(ev) > 1 else float("nan"),
            len(ev),
            100 * ev.sum(),
        )

        with open(out_dir / f"diff_pca_{variant}_{emb_type}.pkl", "wb") as fh:
            pickle.dump(svd, fh)

    out["model_variants"] = np.array(list(data.keys()))
    out_path = out_dir / f"diff_pca_{emb_type}.npz"
    np.savez(out_path, **out)
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
        process_emb_type(emb_type, data, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
