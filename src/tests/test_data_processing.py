import pandas as pd

from src.data_processing import organize_and_cluster


def test_organize_and_cluster_preserves_delta_enrichment_column():
    df_d2 = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "aa": ["AAA", "AAB", "AAC"],
            "num_mut": [2, 2, 2],
            "mut": ["A1B;C2D", "A1B;E3F", "G4H;C2D"],
            "M22_binding_count_adj": [10.0, 8.0, 6.0],
            "M22_non_binding_count_adj": [5.0, 4.0, 3.0],
            "M22_binding_enrichment_adj": [1.0, 0.8, 0.6],
            "delta_M22_binding_enrichment_adj": [0.3, -0.1, 0.05],
        }
    )

    clustered = organize_and_cluster(df_d2, cluster_by=1)

    assert "delta_M22_binding_enrichment_adj" in clustered.columns
    assert "M22_binding_enrichment_adj" not in clustered.columns
    assert "M22_binding_enrichment" in clustered.columns