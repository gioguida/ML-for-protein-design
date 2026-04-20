import sys
import pandas as pd
from pathlib import Path


class test_config:
    def __init__(self):
        self.preview_count = 0
        self.force_rebuild = False
        self.delta_components = ["cross", "wt_anchors", "within_pos", "within_neg"]     #["cross", "wt_anchors", "within_pos", "within_neg"]
        self.gap = 0.5
        self.wt_pairs_frac = 0.1
        self.cross_pairs_frac = 0.2
        self.strong_pos_threshold = 1.0
        self.strong_neg_threshold = -5.0
        self.min_score_margin = 0.3


def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    for p in (str(repo_root), str(src_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


_add_repo_root_to_path()

from src.dataset import default_data_paths, load_dpo_pair_dataframe


def _print_component_stats(pairs_df: pd.DataFrame, preview_count: int) -> None:
    components = pairs_df["delta_component"].unique()
    for comp in sorted(components):
        sub = pairs_df[pairs_df["delta_component"] == comp]
        margins = sub["delta_margin"].astype(float)
        chosen = sub["chosen_delta"].astype(float)
        rejected = sub["rejected_delta"].astype(float)
        print(
            f"\n  [{comp}]  n={len(sub)}\n"
            f"    delta_margin  | mean={margins.mean():.3f}  median={margins.median():.3f}"
            f"  min={margins.min():.3f}  max={margins.max():.3f}\n"
            f"    chosen_delta  | mean={chosen.mean():.3f}  min={chosen.min():.3f}"
            f"  max={chosen.max():.3f}\n"
            f"    rejected_delta| mean={rejected.mean():.3f}  min={rejected.min():.3f}"
            f"  max={rejected.max():.3f}"
        )

        show_n = min(max(0, int(preview_count)), len(sub))
        if show_n == 0:
            continue

        print(f"    Top {show_n} margin pairs:")
        for _, row in sub.nlargest(show_n, "delta_margin").iterrows():
            print(
                f"      margin={float(row['delta_margin']):.3f}"
                f"  chosen={row['chosen_sequence']}"
                f"  rejected={row['rejected_sequence']}"
            )
        print(f"    Bottom {show_n} margin pairs:")
        for _, row in sub.nsmallest(show_n, "delta_margin").iterrows():
            print(
                f"      margin={float(row['delta_margin']):.3f}"
                f"  chosen={row['chosen_sequence']}"
                f"  rejected={row['rejected_sequence']}"
            )


def main() -> None:
    args = test_config()
    paths = default_data_paths()

    pairs_df = load_dpo_pair_dataframe(
        pairing_strategy="delta_based",
        raw_csv_path=paths["raw_m22"],
        processed_dir=paths["processed_dir"],
        force_rebuild=args.force_rebuild,
        delta_components=args.delta_components,
        gap=args.gap,
        wt_pairs_frac=args.wt_pairs_frac,
        cross_pairs_frac=args.cross_pairs_frac,
        strong_pos_threshold=args.strong_pos_threshold,
        strong_neg_threshold=args.strong_neg_threshold,
        min_score_margin=args.min_score_margin,
    )

    if pairs_df.empty:
        print("No pairs produced.")
        return

    margins = pairs_df["delta_margin"].astype(float)
    print(f"Strategy: delta_based (global, no clustering)  |  source_view={pairs_df['source_view'].iloc[0]}")
    print(
        f"Total pairs: {len(pairs_df)}  |  "
        f"margin mean={margins.mean():.3f}  median={margins.median():.3f}"
        f"  min={margins.min():.3f}  max={margins.max():.3f}"
    )
    print("\nPer-component breakdown:")
    _print_component_stats(pairs_df, preview_count=args.preview_count)


if __name__ == "__main__":
    main()
