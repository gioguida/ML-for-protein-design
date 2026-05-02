"""Per-variant OAS UMAP, parameterized by what we color it with.

Asks: did fine-tuning change how the model organizes antibody sequence space along a chosen
biological axis (germline family, J-gene, V-gene-within-family, somatic-hypermutation level,
CDR-H3 length)? UMAP is fitted independently per variant on that variant's OAS whole-VH
embeddings — per-variant projections are NOT comparable across panels.

Reads
-----
- N .npz files written by ``extract_oas_embeddings.py`` (one per variant). Old combined-format
  npz files from ``extract_embeddings.py`` are not accepted.

Writes (200 DPI, in ``{output_dir}/``)
-------------------------------------
Output filename depends on --color-by:
- germline_family       → ``oas_umap_germline_{emb_type}.png``
- j_gene                → ``oas_umap_jgene_{emb_type}.png``
- vgene_within_family   → ``oas_umap_vgene_within_{family}_{emb_type}.png``
- shm_within_family     → ``oas_umap_shm_within_{family}_{emb_type}.png``
- cdr3_length           → ``oas_umap_cdr3len_{emb_type}.png``
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import umap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_oas_embeddings import SCHEMA_VERSION  # noqa: E402

EMB_TYPES = ["whole_seq"]  # cdrh3 has variable lengths in OAS, less interpretable
N_TOP_CATEGORIES = 7
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_oas_umap")


COLOR_BY_CHOICES = [
    "germline_family",
    "j_gene",
    "vgene_within_family",
    "shm_within_family",
    "cdr3_length",
]
WITHIN_FAMILY_MODES = {"vgene_within_family", "shm_within_family"}
CONTINUOUS_MODES = {"shm_within_family", "cdr3_length"}


# --------------------------------------------------------------------------- I/O


def load_variant(path: Path) -> Tuple[str, Dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    if "schema_version" not in z.files:
        raise ValueError(
            f"{path} has no schema_version field; this script expects an OAS-only .npz "
            f"written by extract_oas_embeddings.py. Combined-format files from the older "
            f"extract_embeddings.py pipeline are not supported."
        )
    if str(z["schema_version"][0]) != SCHEMA_VERSION:
        log.warning("%s has schema_version=%s (expected %s) — proceeding, but rebuilding "
                    "is recommended", path, str(z["schema_version"][0]), SCHEMA_VERSION)
    if "v_call" not in z.files:
        raise ValueError(
            f"{path} lacks the OAS metadata fields (v_call, j_call, …) — please regenerate "
            f"with extract_oas_embeddings.py."
        )
    variant = str(z["model_variant"][0])
    return variant, {k: z[k] for k in z.files}


# --------------------------------------------------------------- field selection


def compute_color_field(
    data: Dict[str, np.ndarray], color_by: str, filter_family: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask, values) where mask selects which OAS rows to plot and values is the
    coloring field for the surviving rows.

    For categorical modes, values is a string array. For continuous modes, float.
    """
    n = data["v_call"].shape[0]
    keep = np.ones(n, dtype=bool)

    # Apply within-family filter if applicable.
    if color_by in WITHIN_FAMILY_MODES:
        if not filter_family:
            raise ValueError(f"--filter-family is required for --color-by {color_by}")
        # Permit either family-level (IGHV3) or gene-level (IGHV3-23) filters.
        v_calls = data["v_call"].astype(str)
        v_families = data["v_family"].astype(str)
        if "-" in filter_family:
            keep &= v_calls == filter_family
        else:
            keep &= v_families == filter_family

    if color_by == "germline_family":
        values = data["v_family"].astype(str)
    elif color_by == "j_gene":
        values = data["j_call"].astype(str)
    elif color_by == "vgene_within_family":
        values = data["v_call"].astype(str)
    elif color_by == "shm_within_family":
        v_id = data["v_identity"].astype(np.float32)
        values = 100.0 * (1.0 - v_id)
        keep &= ~np.isnan(values)
    elif color_by == "cdr3_length":
        cdr_len = data["cdr3_len"].astype(np.int32)
        values = cdr_len.astype(np.float32)
        keep &= cdr_len >= 0
    else:
        raise ValueError(f"Unknown --color-by: {color_by}")

    return keep, values[keep]


def get_oas_subset(
    data: Dict[str, np.ndarray], emb_type: str, color_by: str, filter_family: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (oas_emb, color_values) with only valid (non-NaN-embedding, in-filter) rows."""
    keep, values = compute_color_field(data, color_by, filter_family)
    emb = data[f"{emb_type}_embs"][keep]
    valid = ~np.isnan(emb).any(axis=1)
    return emb[valid], values[valid]


# ------------------------------------------------------------- categorical palette


def top_category_palette(values_all: np.ndarray, n_top: int = N_TOP_CATEGORIES) -> Dict[str, tuple]:
    counts = Counter(v for v in values_all if v)
    top = [v for v, _ in counts.most_common(n_top)]
    cmap = plt.get_cmap("tab10")
    palette: Dict[str, tuple] = {v: cmap(i) for i, v in enumerate(top)}
    palette["other"] = (0.55, 0.55, 0.55, 1.0)
    palette[""] = (0.85, 0.85, 0.85, 1.0)  # unknown
    return palette


def categorize(values: np.ndarray, palette: Dict[str, tuple]) -> np.ndarray:
    return np.array([v if v in palette else "other" for v in values])


# ------------------------------------------------------------------------ panels


def make_categorical_panel(ax, coords, cat, palette, title):
    for label, color in palette.items():
        m = cat == label
        if m.any():
            display = "(unknown)" if label == "" else label
            ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.7, c=[color],
                       edgecolors="none", label=display)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=7, frameon=False, markerscale=1.4)


def make_continuous_panel(ax, coords, values, title, vmin, vmax, label):
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.8, c=values,
                    cmap="viridis", vmin=vmin, vmax=vmax, edgecolors="none")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    return sc


def fit_umap(emb: np.ndarray) -> np.ndarray:
    n_neighbors = min(UMAP_N_NEIGHBORS, max(2, len(emb) - 1))
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=UMAP_MIN_DIST, random_state=SEED,
    )
    return reducer.fit_transform(emb)


# ------------------------------------------------------------------ orchestration


def output_filename(color_by: str, filter_family: Optional[str], emb_type: str) -> str:
    if color_by == "germline_family":
        return f"oas_umap_germline_{emb_type}.png"
    if color_by == "j_gene":
        return f"oas_umap_jgene_{emb_type}.png"
    if color_by == "vgene_within_family":
        return f"oas_umap_vgene_within_{filter_family}_{emb_type}.png"
    if color_by == "shm_within_family":
        return f"oas_umap_shm_within_{filter_family}_{emb_type}.png"
    if color_by == "cdr3_length":
        return f"oas_umap_cdr3len_{emb_type}.png"
    raise ValueError(color_by)


def suptitle_for(color_by: str, filter_family: Optional[str], emb_type: str) -> str:
    base = {
        "germline_family": "OAS UMAP — colored by V-gene family",
        "j_gene": "OAS UMAP — colored by J-gene",
        "vgene_within_family": f"OAS UMAP — within {filter_family}, colored by V-gene",
        "shm_within_family": f"OAS UMAP — within {filter_family}, colored by SHM (100−v_identity %)",
        "cdr3_length": "OAS UMAP — colored by CDR-H3 length",
    }[color_by]
    return (f"{base} ({emb_type})\n"
            f"(per-variant fits; UMAP axes NOT comparable across panels)")


def fit_and_plot_categorical(
    variants_data: List[Tuple[str, np.ndarray, np.ndarray]],
    palette: Dict[str, tuple],
    out_path: Path,
    color_by: str,
    filter_family: Optional[str],
    emb_type: str,
):
    n_cols = len(variants_data)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.6), squeeze=False)
    for col, (variant, emb, values) in enumerate(variants_data):
        log.info("[%s] fitting UMAP on %d points", variant, len(emb))
        coords = fit_umap(emb)
        cat = categorize(values, palette)
        make_categorical_panel(axes[0, col], coords, cat, palette, variant)
    fig.suptitle(suptitle_for(color_by, filter_family, emb_type), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def fit_and_plot_continuous(
    variants_data: List[Tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    color_by: str,
    filter_family: Optional[str],
    emb_type: str,
    cbar_label: str,
):
    n_cols = len(variants_data)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.6), squeeze=False)
    all_values = np.concatenate([v for _, _, v in variants_data]) if variants_data else np.array([])
    if all_values.size == 0:
        log.warning("No points to plot for %s — skipping %s", color_by, out_path)
        plt.close(fig)
        return
    vmin = float(np.nanpercentile(all_values, 2))
    vmax = float(np.nanpercentile(all_values, 98))
    if vmin == vmax:
        vmax = vmin + 1.0
    last_sc = None
    for col, (variant, emb, values) in enumerate(variants_data):
        log.info("[%s] fitting UMAP on %d points", variant, len(emb))
        coords = fit_umap(emb)
        last_sc = make_continuous_panel(axes[0, col], coords, values, variant,
                                        vmin, vmax, cbar_label)
    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes[0, :].tolist(), shrink=0.8, pad=0.02)
        cbar.set_label(cbar_label)
    fig.suptitle(suptitle_for(color_by, filter_family, emb_type), fontsize=12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz_files", nargs="+", type=Path,
                   help="OAS-only .npz files from extract_oas_embeddings.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--color-by", required=True, choices=COLOR_BY_CHOICES)
    p.add_argument("--filter-family", default=None,
                   help="V family (e.g. IGHV3) or V gene (e.g. IGHV3-23) to restrict to. "
                        "Required for vgene_within_family / shm_within_family.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.color_by in WITHIN_FAMILY_MODES and not args.filter_family:
        log.error("--filter-family is required when --color-by=%s", args.color_by)
        return 2

    variants_loaded = [load_variant(p) for p in args.npz_files]

    for emb_type in EMB_TYPES:
        per_variant_data: List[Tuple[str, np.ndarray, np.ndarray]] = []
        all_values: List = []
        for variant, data in variants_loaded:
            emb, values = get_oas_subset(data, emb_type, args.color_by, args.filter_family)
            if len(emb) == 0:
                log.warning("Variant %s has 0 OAS rows after filtering for color_by=%s "
                            "(filter_family=%s) — skipping panel.",
                            variant, args.color_by, args.filter_family)
                continue
            per_variant_data.append((variant, emb, values))
            all_values.extend(values.tolist())

        if not per_variant_data:
            log.error("No variants with usable points for color_by=%s — aborting", args.color_by)
            return 1

        out_path = args.output_dir / output_filename(args.color_by, args.filter_family, emb_type)

        if args.color_by in CONTINUOUS_MODES:
            cbar_label = {
                "shm_within_family": "100 × (1 − v_identity)  [%]",
                "cdr3_length": "CDR-H3 length (aa)",
            }[args.color_by]
            fit_and_plot_continuous(per_variant_data, out_path,
                                    args.color_by, args.filter_family, emb_type, cbar_label)
        else:
            palette = top_category_palette(np.array(all_values))
            log.info("Palette for %s (top %d): %s", args.color_by, N_TOP_CATEGORIES,
                     [k for k in palette if k not in ("", "other")])
            fit_and_plot_categorical(per_variant_data, palette, out_path,
                                     args.color_by, args.filter_family, emb_type)

    return 0


if __name__ == "__main__":
    sys.exit(main())
