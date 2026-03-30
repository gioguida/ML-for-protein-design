import pandas as pd
import numpy as np
from pathlib import Path

def get_distance2_data(filename):
    '''Reads the CSV file and filters for entries where num_mut equals 2'''

    df = pd.read_csv(filename)
    df_filtered = df[df['num_mut'] == 2]
    return df_filtered


def d2_stats(df):
    '''Computes and prints statistics for the distance-2 dataset
    including total entries, number of clusters, average sequences per cluster,
    a cluster is defined by the class of sequences sharing the same first mutation (mut1)'''

    # Total entries
    print(f"Total entries: {len(df)}")
    # Split the 'mut' column into 'mut1' and 'mut2'
    muts = df['mut'].str.split(';', expand=True)
    if muts.shape[1] == 2:
        df['mut1'] = muts[0]
        df['mut2'] = muts[1]
        
        # Calculate clusters based on mut1
        clusters = df.groupby('mut1').size()
        print(f"Number of clusters: {len(clusters)}")
        print(f"Average number of sequences per cluster: {clusters.mean():.2f}")
        print(f"Min sequences in a cluster: {clusters.min()}")
        print(f"Max sequences in a cluster: {clusters.max()}")

        # Calculate pairs sharing exactly one mutation (mut1 or mut2)
        pairs_mut1 = sum(n * (n - 1) // 2 for n in clusters)
        pairs_mut2 = sum(n * (n - 1) // 2 for n in df.groupby('mut2').size())
        total_one_shared = pairs_mut1 + pairs_mut2
        print(f"Total pairs sharing exactly one mutation: {total_one_shared}")
    else:
        print("Error: Could not split mutations into exactly two parts.")


def organize_and_save_clusters(df, output_filename='D2_clustered.csv'):
    '''Organizes the distance-2 data into clusters based on the first mutation (mut1),
    calculates a new enrichment score, sorts within clusters by this score'''

    # Ensure mut1 is present
    df['mut1'] = df['mut'].str.split(';').str[0]
    
    # Generate cluster index based on mut1
    unique_clusters = df['mut1'].unique()
    cluster_mapping = {mut: idx for idx, mut in enumerate(unique_clusters)}
    df['cluster_idx'] = df['mut1'].map(cluster_mapping)
    
    # Calculate new score
    bind_count = df['M22_binding_count_adj']
    non_bind_count = df['M22_non_binding_count_adj']
    # Replace zeros with a small number to avoid log(0) if any, or just use np.log since inputs should be positive
    df['M22_binding_enrichment'] = np.log(np.maximum(bind_count, 1e-9)) - np.log(np.maximum(non_bind_count, 1e-9))
    
    # Sort within clusters by the new score in descending order
    df = df.sort_values(by=['cluster_idx', 'M22_binding_enrichment'], ascending=[True, False])
    
    # Rename original enrichment column if it exists and we're replacing it
    if 'M22_binding_enrichment_adj' in df.columns:
        df = df.drop(columns=['M22_binding_enrichment_adj'])
        
    # Keep only the requested columns in the specified order
    ordered_cols = ['Unnamed: 0', 'aa', 'num_mut', 'mut', 'M22_binding_count_adj', 'M22_non_binding_count_adj', 'M22_binding_enrichment', 'cluster_idx']
    
    # Only select columns that actually exist (e.g., Unnamed: 0 might be missing)
    final_cols = [col for col in ordered_cols if col in df.columns]
    df = df[final_cols]

    return df
    

if __name__ == '__main__':
    # extract only distance-2 data if not already done
    df_d2 = get_distance2_data('M22_binding_enrichment.csv')
    df_d2.to_csv('D2.csv', index=False)
    
    # compute and print statistics for distance-2 data
    d2_stats(df_d2)
    
    d2_clustered = organize_and_save_clusters(df_d2)
    d2_clustered.to_csv('D2_clustered.csv', index=False)



