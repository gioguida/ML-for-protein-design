import argparse
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


REQUIRED_RAW_COLUMNS = {
    "aa",
    "num_mut",
    "mut",
    "M22_binding_count_adj",
    "M22_non_binding_count_adj",
}


def _read_raw_data(raw_csv_path: Union[str, Path]) -> pd.DataFrame:
    """Read raw M22 data and validate required columns."""
    raw_csv_path = Path(raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)
    missing_cols = REQUIRED_RAW_COLUMNS.difference(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Raw data file is missing required columns: {missing}")
    return df


def get_distance2_data(raw_input: Union[pd.DataFrame, str, Path]) -> Tuple[pd.DataFrame, float, float]:
    """Extract distance-2 rows and global binding/non-binding totals.

    The totals are computed on the full raw dataset (not only distance-2),
    matching your current enrichment normalization logic.
    """
    df = raw_input if isinstance(raw_input, pd.DataFrame) else _read_raw_data(raw_input)
    df_filtered = df[df["num_mut"] == 2].copy()
    n_bind = float(df["M22_binding_count_adj"].sum())
    n_non_bind = float(df["M22_non_binding_count_adj"].sum())
    return df_filtered, n_bind, n_non_bind


def d2_stats(df_d2: pd.DataFrame) -> Dict[str, float]:
    """Compute summary stats for the distance-2 dataset."""
    df = df_d2.copy()
    stats: Dict[str, float] = {"total_entries": float(len(df))}

    muts = df["mut"].str.split(";", expand=True)
    if muts.shape[1] != 2:
        raise ValueError("Could not split mutations into exactly two parts.")

    df["mut1"] = muts[0]
    df["mut2"] = muts[1]

    clusters_mut1 = df.groupby("mut1").size()
    clusters_mut2 = df.groupby("mut2").size()

    pairs_mut1 = int(sum(n * (n - 1) // 2 for n in clusters_mut1))
    pairs_mut2 = int(sum(n * (n - 1) // 2 for n in clusters_mut2))

    stats.update(
        {
            "num_clusters_mut1": float(len(clusters_mut1)),
            "avg_cluster_size_mut1": float(clusters_mut1.mean()),
            "min_cluster_size_mut1": float(clusters_mut1.min()),
            "max_cluster_size_mut1": float(clusters_mut1.max()),
            "pairs_sharing_one_mutation": float(pairs_mut1 + pairs_mut2),
        }
    )
    return stats


def _compute_m22_binding_enrichment(
    bind_count_adj: pd.Series,
    non_bind_count_adj: pd.Series,
    n_bind: float,
    n_non_bind: float,
) -> pd.Series:
    """Compute log2 enrichment with a small epsilon for numerical stability."""
    eps = 1e-12
    return (
        np.log2(bind_count_adj.clip(lower=eps))
        - np.log2(non_bind_count_adj.clip(lower=eps))
        - np.log2(max(n_bind, eps))
        + np.log2(max(n_non_bind, eps))
    )


def organize_and_cluster(
    df_d2: pd.DataFrame,
    cluster_by: int = 1,
    n_bind: float = None,
    n_non_bind: float = None,
) -> pd.DataFrame:
    """Cluster distance-2 data by first or second mutation and sort by enrichment."""
    if cluster_by not in (1, 2):
        raise ValueError("cluster_by must be 1 or 2")

    df = df_d2.copy()
    mut_idx = cluster_by - 1
    target_mut = f"mut{cluster_by}"

    mut_parts = df["mut"].str.split(";", expand=True)
    if mut_parts.shape[1] != 2:
        raise ValueError("Could not split mutations into exactly two parts.")
    df[target_mut] = mut_parts[mut_idx]

    unique_clusters = df[target_mut].unique()
    cluster_mapping = {mut: idx for idx, mut in enumerate(unique_clusters)}
    df["cluster_idx"] = df[target_mut].map(cluster_mapping)

    bind_count_adj = df["M22_binding_count_adj"]
    non_bind_count_adj = df["M22_non_binding_count_adj"]
    n_binding = n_bind if n_bind is not None else float(bind_count_adj.sum())
    n_non_binding = n_non_bind if n_non_bind is not None else float(non_bind_count_adj.sum())

    df["M22_binding_enrichment"] = _compute_m22_binding_enrichment(
        bind_count_adj=bind_count_adj,
        non_bind_count_adj=non_bind_count_adj,
        n_bind=n_binding,
        n_non_bind=n_non_binding,
    )

    df = df.sort_values(
        by=["cluster_idx", "M22_binding_enrichment"],
        ascending=[True, False],
    )

    if "M22_binding_enrichment_adj" in df.columns:
        df = df.drop(columns=["M22_binding_enrichment_adj"])

    ordered_cols = [
        "Unnamed: 0",
        "aa",
        "num_mut",
        "mut",
        "M22_binding_count_adj",
        "M22_non_binding_count_adj",
        "M22_binding_enrichment",
        "cluster_idx",
    ]
    final_cols = [col for col in ordered_cols if col in df.columns]
    return df[final_cols]


def build_processed_views(
    raw_csv_path: Union[str, Path],
    processed_dir: Union[str, Path],
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Build D2 and cluster views from raw data if missing or stale.

    Returns paths to:
    - D2.csv
    - D2_clustered_mut1.csv
    - D2_clustered_mut2.csv
    """
    raw_csv_path = Path(raw_csv_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "d2": processed_dir / "D2.csv",
        "d2_clustered_mut1": processed_dir / "D2_clustered_mut1.csv",
        "d2_clustered_mut2": processed_dir / "D2_clustered_mut2.csv",
    }

    raw_mtime = raw_csv_path.stat().st_mtime
    all_outputs_exist = all(path.exists() for path in output_paths.values())
    outputs_are_fresh = all(
        path.stat().st_mtime >= raw_mtime for path in output_paths.values() if path.exists()
    )

    should_rebuild = force or (not all_outputs_exist) or (not outputs_are_fresh)
    if not should_rebuild:
        if verbose:
            print("Processed views are up to date. Reusing existing files.")
        return output_paths

    if verbose:
        print("Building processed views from raw M22 data...")

    raw_df = _read_raw_data(raw_csv_path)
    df_d2, n_bind, n_non_bind = get_distance2_data(raw_df)

    output_paths["d2"].parent.mkdir(parents=True, exist_ok=True)
    df_d2.to_csv(output_paths["d2"], index=False)

    d2_clustered_mut1 = organize_and_cluster(
        df_d2,
        cluster_by=1,
        n_bind=n_bind,
        n_non_bind=n_non_bind,
    )
    d2_clustered_mut1.to_csv(output_paths["d2_clustered_mut1"], index=False)

    d2_clustered_mut2 = organize_and_cluster(
        df_d2,
        cluster_by=2,
        n_bind=n_bind,
        n_non_bind=n_non_bind,
    )
    d2_clustered_mut2.to_csv(output_paths["d2_clustered_mut2"], index=False)

    if verbose:
        stats = d2_stats(df_d2)
        print(
            "D2 stats: "
            f"entries={int(stats['total_entries'])}, "
            f"clusters(mut1)={int(stats['num_clusters_mut1'])}, "
            f"avg_cluster_size(mut1)={stats['avg_cluster_size_mut1']:.2f}"
        )
        print(f"Wrote: {output_paths['d2']}")
        print(f"Wrote: {output_paths['d2_clustered_mut1']}")
        print(f"Wrote: {output_paths['d2_clustered_mut2']}")

    return output_paths


def load_processed_views(
    raw_csv_path: Union[str, Path],
    processed_dir: Union[str, Path],
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Ensure processed views exist, then load them into memory."""
    paths = build_processed_views(raw_csv_path, processed_dir, force=force, verbose=False)
    return {
        "d2": pd.read_csv(paths["d2"]),
        "d2_clustered_mut1": pd.read_csv(paths["d2_clustered_mut1"]),
        "d2_clustered_mut2": pd.read_csv(paths["d2_clustered_mut2"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build processed D2 views from raw M22 data.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("../data/raw/M22_binding_enrichment.csv"),
        help="Path to raw M22 CSV.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("../data/processed"),
        help="Directory to store processed CSVs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even if processed files are fresh.",
    )
    args = parser.parse_args()

    build_processed_views(
        raw_csv_path=args.raw,
        processed_dir=args.processed_dir,
        force=args.force,
        verbose=True,
    )



