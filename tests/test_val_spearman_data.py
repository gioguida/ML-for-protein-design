import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _build_cfg(
    train_frac: float,
    val_frac: float,
    test_frac: float,
    min_positive_delta: float,
    n_val_pos: int,
    n_val_neg: int,
    hamming_distance: int,
    stratify_bins: int,
    force_rebuild: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            force_rebuild=bool(force_rebuild),
            train_frac=float(train_frac),
            val_frac=float(val_frac),
            test_frac=float(test_frac),
            min_positive_delta=float(min_positive_delta),
            split=SimpleNamespace(
                hamming_distance=int(hamming_distance),
                stratify_bins=int(stratify_bins),
            ),
            val=SimpleNamespace(
                n_val_pos=int(n_val_pos),
                n_val_neg=int(n_val_neg),
            ),
        )
    )


def _print_summary(label: str, df: pd.DataFrame) -> None:
    print(f"\n[{label}]")
    print(f"rows: {len(df)}")

    if len(df) == 0:
        return

    if "num_mut" in df.columns:
        counts = (
            pd.to_numeric(df["num_mut"], errors="coerce")
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        print(f"num_mut counts: {counts}")

    if "M22_binding_enrichment_adj" in df.columns:
        enr = pd.to_numeric(df["M22_binding_enrichment_adj"], errors="coerce").dropna()
        if len(enr) > 0:
            print(
                "enrichment stats: "
                f"min={enr.min():.4f}, q25={enr.quantile(0.25):.4f}, "
                f"median={enr.median():.4f}, q75={enr.quantile(0.75):.4f}, max={enr.max():.4f}"
            )

    if "aa" in df.columns:
        print(f"unique sequences (aa): {df['aa'].astype(str).nunique()}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from protein_design.dpo.data_processing import build_validation_perplexity_csvs

    raw_csv = root / "data" / "raw" / "ED2_M22_binding_enrichment.csv"
    processed_dir = root / "data" / "processed"
    seed = 42
    force = True
    train_frac = 0.8
    val_frac = 0.1
    test_frac = 0.1
    min_positive_delta = 3.0
    n_val_pos = 200
    n_val_neg = 400
    hamming_distance = 1
    stratify_bins = 10

    cfg = _build_cfg(
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        min_positive_delta=min_positive_delta,
        n_val_pos=n_val_pos,
        n_val_neg=n_val_neg,
        hamming_distance=hamming_distance,
        stratify_bins=stratify_bins,
        force_rebuild=force,
    )

    outputs = build_validation_perplexity_csvs(
        raw_csv_path=raw_csv,
        processed_dir=processed_dir,
        cfg=cfg,
        seed=seed,
        force=force,
        verbose=False,
    )

    val_spearman_path = outputs["val_spearman"]
    val_pos_path = outputs["val_pos"]
    val_neg_path = outputs["val_neg"]

    val_spearman = pd.read_csv(val_spearman_path)
    val_pos = pd.read_csv(val_pos_path)
    val_neg = pd.read_csv(val_neg_path)

    print("Built validation files:")
    print(f"- val_spearman: {val_spearman_path}")
    print(f"- val_pos:      {val_pos_path}")
    print(f"- val_neg:      {val_neg_path}")

    _print_summary("val_spearman (full val split)", val_spearman)
    _print_summary("val_pos (delta-filtered)", val_pos)
    _print_summary("val_neg (delta-filtered)", val_neg)

    if "aa" in val_spearman.columns and "aa" in val_pos.columns and "aa" in val_neg.columns:
        spearman_aas = set(val_spearman["aa"].astype(str))
        ppl_aas = set(val_pos["aa"].astype(str)).union(set(val_neg["aa"].astype(str)))
        only_spearman = len(spearman_aas - ppl_aas)
        overlap = len(spearman_aas & ppl_aas)
        print("\n[coverage]")
        print(f"spearman-only sequences: {only_spearman}")
        print(f"overlap with pos/neg:    {overlap}")


if __name__ == "__main__":
    main()
