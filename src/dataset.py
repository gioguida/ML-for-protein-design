"""Dataset loading helpers for DPO workflow.

This module integrates preprocessing so callers can rely on raw data only.
If processed D2 files are missing or stale, they are rebuilt automatically.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from .data_processing import build_processed_views


def _project_root() -> Path:
	"""Return the repository root path (code/)."""
	return Path(__file__).resolve().parents[1]


def default_data_paths() -> Dict[str, Path]:
	"""Return default raw and processed data paths."""
	root = _project_root()
	return {
		"raw_m22": root / "data" / "raw" / "M22_binding_enrichment.csv",
		"processed_dir": root / "data" / "processed",
	}


def ensure_processed_data(
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
	verbose: bool = False,
) -> Dict[str, Path]:
	"""Ensure D2 processed views exist and return their paths."""
	defaults = default_data_paths()
	raw_csv_path = defaults["raw_m22"] if raw_csv_path is None else Path(raw_csv_path)
	processed_dir = (
		defaults["processed_dir"] if processed_dir is None else Path(processed_dir)
	)

	return build_processed_views(
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force=force_rebuild,
		verbose=verbose,
	)


def load_distance2_dataframe(
	view: str = "mut1",
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
) -> pd.DataFrame:
	"""Load one distance-2 dataframe view.

	Args:
		view:
			- "base"  -> D2.csv
			- "mut1"  -> D2_clustered_mut1.csv
			- "mut2"  -> D2_clustered_mut2.csv
	"""
	paths = ensure_processed_data(
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force_rebuild=force_rebuild,
		verbose=False,
	)

	view_to_key = {
		"base": "d2",
		"mut1": "d2_clustered_mut1",
		"mut2": "d2_clustered_mut2",
	}
	if view not in view_to_key:
		raise ValueError("view must be one of: base, mut1, mut2")

	return pd.read_csv(paths[view_to_key[view]])


def load_all_distance2_dataframes(
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
) -> Dict[str, pd.DataFrame]:
	"""Load base and clustered D2 dataframes."""
	paths = ensure_processed_data(
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force_rebuild=force_rebuild,
		verbose=False,
	)

	return {
		"d2": pd.read_csv(paths["d2"]),
		"d2_clustered_mut1": pd.read_csv(paths["d2_clustered_mut1"]),
		"d2_clustered_mut2": pd.read_csv(paths["d2_clustered_mut2"]),
	}

