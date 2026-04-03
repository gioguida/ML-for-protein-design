import pandas as pd

from src.dataset import build_dpo_pairs_from_clustered_dataframe


def _example_cluster_df() -> pd.DataFrame:
	return pd.DataFrame(
		{
			"Unnamed: 0": [10, 11, 12, 13, 14, 15],
			"aa": ["SEQ_B", "SEQ_F", "SEQ_A", "SEQ_E", "SEQ_D", "SEQ_C"],
			"mut": ["mB", "mF", "mA", "mE", "mD", "mC"],
			"cluster_idx": [0, 0, 0, 0, 0, 0],
			"delta_M22_binding_enrichment_adj": [0.6, -0.4, 0.9, -0.3, -0.2, -0.1],
		}
	)


def test_build_dpo_pairs_positive_vs_tail():
	clustered = _example_cluster_df()
	pairs = build_dpo_pairs_from_clustered_dataframe(
		clustered_df=clustered,
		pairing_strategy="positive_vs_tail",
		min_positive_delta=0.0,
		source_view="mut1",
	)

	assert len(pairs) == 2
	assert pairs.loc[0, "chosen_sequence"] == "SEQ_A"
	assert pairs.loc[0, "rejected_sequence"] == "SEQ_F"
	assert pairs.loc[1, "chosen_sequence"] == "SEQ_B"
	assert pairs.loc[1, "rejected_sequence"] == "SEQ_E"
	assert (pairs["delta_margin"] > 0).all()


def test_build_dpo_pairs_positive_only_extremes():
	clustered = _example_cluster_df()
	pairs = build_dpo_pairs_from_clustered_dataframe(
		clustered_df=clustered,
		pairing_strategy="positive_only_extremes",
		min_positive_delta=0.0,
		source_view="mut2",
	)

	assert len(pairs) == 2
	assert pairs.loc[0, "chosen_sequence"] == "SEQ_A"
	assert pairs.loc[0, "rejected_sequence"] == "SEQ_E"
	assert pairs.loc[1, "chosen_sequence"] == "SEQ_B"
	assert pairs.loc[1, "rejected_sequence"] == "SEQ_F"
	assert (pairs["delta_margin"] > 0).all()


def test_build_dpo_pairs_rejects_invalid_strategy():
	clustered = _example_cluster_df()
	try:
		build_dpo_pairs_from_clustered_dataframe(
			clustered_df=clustered,
			pairing_strategy="not_a_strategy",  # type: ignore[arg-type]
		)
		assert False, "Expected ValueError for invalid pairing strategy"
	except ValueError as exc:
		assert "pairing_strategy" in str(exc)