import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from utils import WILD_TYPE, PairTuple
import warnings

class test_config:
    def __init__(self):
        self.pairing_strategy = "delta_based"  # "positive_vs_tail", "positive_only_extremes", "both_structured", "delta_based"
        self.preview_count = 0
        self.include_views = ("mut1", "mut2")
        self.force_rebuild = False
        self.min_positive_delta = 3.0
        self.min_delta_margin = 5.0
        self.gap = 0.5
        self.wt_pairs_frac = 0.1
        self.cross_pairs_frac = 0.1
        self.strong_pos_threshold = 1.0
        self.strong_neg_threshold = -5.0
        self.min_score_margin = 0.1
        self.deduplicate_across_views = True

def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_add_repo_root_to_path()

from src.dataset import default_data_paths, load_dpo_pair_dataframe

def _pair_delta_based(
    sequences_df: pd.DataFrame,
    delta_col: str,
    seq_col: str,
    gap: float,
    wt_pairs_frac: float,
    strong_pos_threshold: float = 1.0,
    strong_neg_threshold: float = -5.0,
) -> List[PairTuple]:
    """
    Pair sequences for weighted DPO training.

    Pairs are created in two ways:
      1. Variant-vs-variant: the sorted list is split into a top half (winners)
         and a bottom half (losers). Each winner is paired with a loser at a
         controlled offset within the bottom half, set by `gap`.
         gap=0.0 -> pair top[i] with bot[i]   (smallest score difference)
         gap=1.0 -> pair top[i] with bot[-1]  (largest score difference)

      2. WT-anchored pairs: a fraction of pairs use the wild type as one member.
         We use clearly positive sequences (winner vs WT) and clearly negative
         sequences (WT vs loser), avoiding ambiguous near-zero sequences that
         would produce near-zero training weights.
    """
    if not 0.0 <= float(gap) <= 1.0:
        raise ValueError("gap must be in [0.0, 1.0].")
    if not 0.0 <= float(wt_pairs_frac) <= 1.0:
        raise ValueError("wt_pairs_frac must be in [0.0, 1.0].")

    # Sort descending by delta score (best binders first).
    sorted_df = sequences_df.sort_values(by=delta_col, ascending=False).reset_index(drop=True)
    n = len(sorted_df)
    if n <= 1:
        return []

    # We need an even number of sequences for clean halving.
    if n % 2 != 0:
        sorted_df = sorted_df.iloc[:-1].reset_index(drop=True)
        n -= 1
    if n <= 1:
        return []

    # ------------------------------------------------------------------ #
    # Part 1: Variant-vs-variant pairs
    # Split sorted list into top half (winners) and bottom half (losers).
    # ------------------------------------------------------------------ #
    half = n // 2
    top_half = sorted_df.iloc[:half].reset_index(drop=True)   # high scores
    bot_half = sorted_df.iloc[half:].reset_index(drop=True)   # low scores

    # gap controls how far into the bottom half each winner reaches.
    # offset=0   -> top[i] paired with bot[i]       (close scores, small signal)
    # offset=max -> top[i] paired with bot[i + max] (far scores, large signal)
    max_offset = half - 1
    offset = int(float(gap) * max_offset)

    variant_pairs: List[PairTuple] = []
    for i in range(half):
        # Clamp so we never go out of bounds at the end of the list.
        j = min(i + offset, half - 1)

        winner_row = top_half.iloc[i]
        loser_row  = bot_half.iloc[j]

        winner = {"aa": winner_row[seq_col], "score": float(winner_row[delta_col])}
        loser  = {"aa": loser_row[seq_col],  "score": float(loser_row[delta_col])}
        variant_pairs.append((winner, loser))

    # ------------------------------------------------------------------ #
    # Part 2: WT-anchored pairs
    # Use strong positives and strong negatives to avoid near-zero noise.
    # ------------------------------------------------------------------ #
    num_wt_pairs  = int(float(wt_pairs_frac) * n)
    num_pos_wt    = num_wt_pairs // 2         # (positive variant) > WT
    num_wt_neg    = num_wt_pairs - num_pos_wt  # WT > (negative variant)

    wt = {"aa": WILD_TYPE, "score": 0.0}
    wt_pairs: List[PairTuple] = []

    if num_wt_pairs > 0:
        # Strong positives: clearly better than WT, not just marginally above zero.
        strong_positives = sorted_df[sorted_df[delta_col].astype(float) > strong_pos_threshold]
        if len(strong_positives) == 0:
            warnings.warn(
                f"No sequences above strong_pos_threshold={strong_pos_threshold}. "
                "Consider lowering the threshold."
            )
        else:
            # Sample randomly so we don't always pick the same top sequences.
            sampled_pos = strong_positives.sample(
                n=min(num_pos_wt, len(strong_positives)),
                replace=False,
            )
            for _, row in sampled_pos.iterrows():
                winner = {"aa": row[seq_col], "score": float(row[delta_col])}
                wt_pairs.append((winner, wt))

        # Strong negatives: clearly worse than WT, not just marginally below zero.
        strong_negatives = sorted_df[sorted_df[delta_col].astype(float) < strong_neg_threshold]
        if len(strong_negatives) == 0:
            warnings.warn(
                f"No sequences below strong_neg_threshold={strong_neg_threshold}. "
                "Consider raising the threshold."
            )
        else:
            sampled_neg = strong_negatives.sample(
                n=min(num_wt_neg, len(strong_negatives)),
                replace=False,
            )
            for _, row in sampled_neg.iterrows():
                loser = {"aa": row[seq_col], "score": float(row[delta_col])}
                wt_pairs.append((wt, loser))

    all_pairs = variant_pairs + wt_pairs

    # ------------------------------------------------------------------ #
    # Sanity check: print a short summary so you can verify the pairing.
    # ------------------------------------------------------------------ #
    if len(all_pairs) > 0:
        score_gaps = [
            abs(w["score"] - l["score"]) for w, l in all_pairs
        ]
        print(
            f"[pairing] Total pairs: {len(all_pairs)} "
            f"({len(variant_pairs)} variant-vs-variant, {len(wt_pairs)} WT-anchored)\n"
            f"[pairing] Score gap  — mean: {np.mean(score_gaps):.2f}, "
            f"min: {np.min(score_gaps):.2f}, max: {np.max(score_gaps):.2f}"
        )

    return all_pairs


def main() -> None:
    args = test_config()
    paths = default_data_paths()
    pairs_df = load_dpo_pair_dataframe(
        pairing_strategy=args.pairing_strategy,
        include_views=args.include_views,
        raw_csv_path=paths["raw_m22"],
        processed_dir=paths["processed_dir"],
        force_rebuild=args.force_rebuild,
        min_positive_delta=args.min_positive_delta,
        min_delta_margin=args.min_delta_margin,
        gap=args.gap,
        wt_pairs_frac=args.wt_pairs_frac,
        cross_pairs_frac=args.cross_pairs_frac,
        strong_pos_threshold=args.strong_pos_threshold,
        strong_neg_threshold=args.strong_neg_threshold,
        min_score_margin=args.min_score_margin,
        deduplicate_across_views=args.deduplicate_across_views,
    )

    if pairs_df.empty:
        print("No pairs available after preprocessing.")
        return
    
    print(f"Pairing strategy: {args.pairing_strategy}")
    print(f"min_positive_delta: {args.min_positive_delta} | min_delta_margin: {args.min_delta_margin}")

    margins = pairs_df["delta_margin"].astype(float)
    print(
        f"Pair stats | n={len(pairs_df)} | "
        f"margin mean={margins.mean():.4f} median={margins.median():.4f} "
        f"min={margins.min():.4f} max={margins.max():.4f}"
    )

    by_view = pairs_df.groupby("source_view").size().sort_values(ascending=False)
    print("Pairs per view: " + ", ".join(f"{k}:{int(v)}" for k, v in by_view.items()))

    show_n = min(max(0, int(args.preview_count)), len(pairs_df))
    if show_n == 0:
        return

    top_examples = pairs_df.nlargest(show_n, "delta_margin")
    low_examples = pairs_df.nsmallest(show_n, "delta_margin")

    print("Top margin examples:")
    for _, row in top_examples.iterrows():
        print(
            f"  view={row['source_view']} cluster={row['cluster_idx']} "
            f"margin={float(row['delta_margin']):.4f} "
            f"chosen={row['chosen_sequence']} rejected={row['rejected_sequence']}"
        )

    print("Bottom margin examples:")
    for _, row in low_examples.iterrows():
        print(
            f"  view={row['source_view']} cluster={row['cluster_idx']} "
            f"margin={float(row['delta_margin']):.4f} "
            f"chosen={row['chosen_sequence']} rejected={row['rejected_sequence']}"
        )


if __name__ == "__main__":
    main()


