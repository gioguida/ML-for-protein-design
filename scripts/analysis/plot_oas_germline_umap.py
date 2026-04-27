"""Per-variant OAS UMAP colored by V-gene germline family.

Asks: did fine-tuning (especially evotuning on OAS) change how the model
clusters antibody germline families? UMAP is fitted independently per variant
on that variant's OAS whole-VH embeddings — per-variant projections are NOT
comparable across panels (UMAP coordinates have no shared meaning).

Reads
-----
- N .npz files written by ``extract_embeddings.py`` (one per variant). The
  ``v_family`` column is preferred. If absent (legacy .npz from before the
  v_family extension was added), we fall back to reproducing the deterministic
  OAS reservoir sample with seed=42 and looking up v_call from the metadata.

Writes (200 DPI, in ``{output_dir}/``)
-------------------------------------
- ``oas_umap_germline_{emb_type}.png`` — one panel per variant, OAS sequences
  only, colored by V-gene family (top families + "other"). Each panel shows
  its own UMAP fit; axes/distances NOT comparable across panels.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from Bio import SeqIO

# Importing the local helper (sibling module) — these scripts run as standalone
# CLIs, so we add the script directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_embeddings import (  # noqa: E402
    DEFAULT_OAS_FASTA,
    DEFAULT_OAS_META,
    SEED,
    _v_family_from_call,
    reservoir_sample_fasta,
)

EMB_TYPES = ["whole_seq"]  # cdrh3 has variable lengths in OAS, less interpretable
N_TOP_FAMILIES = 7  # IGHV1..IGHV7 typically dominate
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_oas_germline_umap")


def load_variant(path: Path) -> Tuple[str, Dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    variant = str(z["model_variant"][0])
    return variant, {k: z[k] for k in z.files}


def recompute_v_family(
    oas_fasta: Path, oas_meta: Path, max_oas: int
) -> Tuple[List[str], np.ndarray]:
    """Reproduce extract_embeddings.py's deterministic OAS reservoir sample
    and look up V-family for each sampled seq_id. Used as a fallback when
    the .npz file lacks a ``v_family`` column.

    Returns
    -------
    seqs : list of full VH strings in sampling order
    families : (n_oas,) array of V-family strings ('' if unknown)
    """
    log.info("Reservoir-sampling %d OAS sequences (seed=%d) for v_family lookup",
             max_oas, SEED)
    pairs = reservoir_sample_fasta(oas_fasta, max_oas)
    seq_ids = [sid for sid, _ in pairs]
    seqs = [s for _, s in pairs]
    wanted = set(seq_ids)
    family_lookup: Dict[str, str] = {}
    cols = ["seq_id", "v_call"]
    for chunk in pd.read_csv(oas_meta, usecols=cols, chunksize=500_000):
        hit = chunk[chunk["seq_id"].isin(wanted)]
        for sid, vcall in zip(hit["seq_id"].astype(str), hit["v_call"]):
            family_lookup[sid] = _v_family_from_call(vcall)
    families = np.array([family_lookup.get(sid, "") for sid in seq_ids])
    return seqs, families


def get_oas_subset(
    data: Dict[str, np.ndarray],
    emb_type: str,
    fallback_families: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (oas_emb, oas_v_family) with only valid (non-NaN) OAS rows."""
    src = data["source_labels"]
    is_oas = src == "oas"

    if "v_family" in data:
        families = data["v_family"][is_oas]
    elif fallback_families is not None:
        if len(fallback_families) != int(is_oas.sum()):
            raise ValueError(
                f"Fallback families length {len(fallback_families)} does not match "
                f"OAS row count {int(is_oas.sum())}. Re-extract embeddings with the "
                "same --max-oas or pass an updated metadata path."
            )
        families = fallback_families
    else:
        raise ValueError("No v_family in npz and no fallback provided.")

    emb = data[f"{emb_type}_embs"][is_oas]
    valid = ~np.isnan(emb).any(axis=1)
    return emb[valid], families[valid]


def family_palette(families: np.ndarray) -> Dict[str, tuple]:
    counts = Counter(f for f in families if f)
    top = [f for f, _ in counts.most_common(N_TOP_FAMILIES)]
    cmap = plt.get_cmap("tab10")
    palette: Dict[str, tuple] = {f: cmap(i) for i, f in enumerate(top)}
    palette["other"] = (0.55, 0.55, 0.55, 1.0)
    palette[""] = (0.85, 0.85, 0.85, 1.0)  # unknown
    return palette


def categorize(families: np.ndarray, palette: Dict[str, tuple]) -> np.ndarray:
    return np.array([f if f in palette else "other" for f in families])


def make_panel(ax, coords, fam_cat, palette, title):
    for label, color in palette.items():
        m = fam_cat == label
        if m.any():
            display = "(unknown)" if label == "" else label
            ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.7, c=[color],
                       edgecolors="none", label=display)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=7, frameon=False, markerscale=1.4)


def fit_and_plot(
    variants_data: List[Tuple[str, np.ndarray, np.ndarray]],
    palette: Dict[str, tuple],
    out_path: Path,
    emb_type: str,
):
    n_cols = len(variants_data)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.6), squeeze=False)
    for col, (variant, emb, families) in enumerate(variants_data):
        n_neighbors = min(UMAP_N_NEIGHBORS, max(2, len(emb) - 1))
        log.info("[%s] fitting UMAP on %d points (n_neighbors=%d)",
                 variant, len(emb), n_neighbors)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=UMAP_MIN_DIST,
            random_state=SEED,
        )
        coords = reducer.fit_transform(emb)
        cat = categorize(families, palette)
        make_panel(axes[0, col], coords, cat, palette, variant)
    fig.suptitle(
        f"OAS UMAP — colored by V-gene family ({emb_type})\n"
        f"(per-variant fits; UMAP axes NOT comparable across panels)",
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
    p.add_argument("npz_files", nargs="+", type=Path,
                   help="Per-variant .npz files from extract_embeddings.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--oas-fasta", type=Path, default=Path(DEFAULT_OAS_FASTA),
                   help="Used only as fallback when v_family is missing from .npz")
    p.add_argument("--oas-meta", type=Path, default=Path(DEFAULT_OAS_META))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants_loaded = [load_variant(p) for p in args.npz_files]

    # Decide whether to use the in-npz v_family or recompute.
    needs_fallback = any("v_family" not in d for _, d in variants_loaded)
    fallback_families = None
    if needs_fallback:
        log.warning("One or more .npz files lack a v_family column — recomputing "
                    "from OAS metadata (deterministic with seed=%d).", SEED)
        # All variants use the same deterministic OAS sample, so one recompute
        # suffices. We still verify length matches each variant downstream.
        n_oas = int((variants_loaded[0][1]["source_labels"] == "oas").sum())
        _, fallback_families = recompute_v_family(args.oas_fasta, args.oas_meta, n_oas)

    for emb_type in EMB_TYPES:
        per_variant_data: List[Tuple[str, np.ndarray, np.ndarray]] = []
        all_families: List[str] = []
        for variant, data in variants_loaded:
            emb, families = get_oas_subset(data, emb_type, fallback_families)
            per_variant_data.append((variant, emb, families))
            all_families.extend(families.tolist())
        palette = family_palette(np.array(all_families))
        log.info("V-family palette (top %d): %s",
                 N_TOP_FAMILIES, [k for k in palette if k not in ("", "other")])
        fit_and_plot(per_variant_data, palette,
                     args.output_dir / f"oas_umap_germline_{emb_type}.png",
                     emb_type)

    return 0


if __name__ == "__main__":
    sys.exit(main())
