#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tennis Player Clustering Project
Clusters professional tennis players by serve and rally metrics using DBSCAN, OPTICS, and Hierarchical clustering.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore[import-untyped]
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def generate_synthetic_data(n_players=15, seed=42):
    """Generate synthetic tennis player data with varied playing styles.
    
    Creates three archetypes: power servers, baseliners, and all-rounders.
    """
    np.random.seed(seed)
    players = []
    
    names = [
        "Rafael Nadal", "Novak Djokovic", "Roger Federer", "Daniil Medvedev",
        "Alexander Zverev", "Carlos Alcaraz", "Stefanos Tsitsipas", "Andrey Rublev",
        "Jannik Sinner", "Taylor Fritz", "John Isner", "Marin Cilic",
        "Diego Schwartzman", "Andy Murray", "Stan Wawrinka"
    ]
    
    for i in range(n_players):
        # Assign playing style
        if i < 5:  # Power servers
            serve_speed = np.random.uniform(205, 230)
            ace_pct = np.random.uniform(0.14, 0.25)
            avg_rally = np.random.uniform(7.0, 9.5)
            win_ratio = np.random.uniform(0.65, 0.75)
        elif i < 10:  # Baseliners
            serve_speed = np.random.uniform(175, 200)
            ace_pct = np.random.uniform(0.06, 0.12)
            avg_rally = np.random.uniform(11.0, 14.0)
            win_ratio = np.random.uniform(0.72, 0.85)
        else:  # All-rounders
            serve_speed = np.random.uniform(185, 205)
            ace_pct = np.random.uniform(0.10, 0.15)
            avg_rally = np.random.uniform(9.5, 11.5)
            win_ratio = np.random.uniform(0.73, 0.82)
        
        # Common variations
        df_pct = np.random.uniform(0.03, 0.08)
        first_serve_in = np.random.uniform(0.55, 0.70)
        first_serve_won = np.random.uniform(0.68, 0.80)
        break_saved = np.random.uniform(0.65, 0.75)
        height = np.random.randint(175, 210)
        
        handedness = np.random.choice(["Right", "Left", "Right"], p=[0.85, 0.10, 0.05])
        surface = np.random.choice(["Hard", "Clay", "Grass"], p=[0.50, 0.35, 0.15])
        
        players.append({
            'player': names[i] if i < len(names) else f"Player_{i+1}",
            'handedness': handedness,
            'dominant_surface': surface,
            'serve_speed_avg': serve_speed,
            'ace_pct': ace_pct,
            'double_fault_pct': df_pct,
            'first_serve_in_pct': first_serve_in,
            'first_serve_points_won_pct': first_serve_won,
            'break_points_saved_pct': break_saved,
            'avg_rally_length': avg_rally,
            'win_ratio': win_ratio,
            'height_cm': height
        })
    
    return pd.DataFrame(players)


def load_data(input_path):
    """Load tennis player data from CSV or generate synthetic data if missing."""
    if os.path.exists(input_path):
        print(f"Loading data from {input_path}")
        return pd.read_csv(input_path)
    else:
        print(f"File not found: {input_path}. Generating synthetic data...")
        df = generate_synthetic_data()
        # Save generated data
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        df.to_csv(input_path, index=False)
        print(f"Generated data saved to {input_path}")
        return df


def prepare_features(df):
    """Extract and normalize numeric features for clustering.
    
    Returns:
        X_scaled: Normalized feature matrix
        feature_names: List of feature column names
    """
    numeric_features = [
        'serve_speed_avg', 'ace_pct', 'double_fault_pct', 'first_serve_in_pct',
        'first_serve_points_won_pct', 'break_points_saved_pct',
        'avg_rally_length', 'win_ratio', 'height_cm'
    ]
    
    X = df[numeric_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Features: {numeric_features}")
    print(f"Data shape: {X_scaled.shape}")
    
    return X_scaled, numeric_features


def cluster_dbscan(X, eps_range=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0], min_samples_range=[2, 3, 4, 5]):
    """DBSCAN clustering: density-based algorithm detecting arbitrarily shaped clusters.
    
    Parameters:
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum samples in a neighborhood to form a core point
    
    Returns best model and hyperparameters based on silhouette score (excluding noise).
    """
    best_score = -1
    best_model = None
    best_params = None
    
    print("\n=== DBSCAN Grid Search ===")
    for eps in eps_range:
        for min_samples in min_samples_range:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1:
                # Compute silhouette excluding noise points
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                    print(f"eps={eps:.1f}, min_samples={min_samples}: "
                          f"n_clusters={n_clusters}, noise={n_noise}, silhouette={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = {'eps': eps, 'min_samples': min_samples}
    
    if best_model is None:
        # Fallback to conservative parameters for small datasets
        print("No valid clusters found with grid search. Using fallback parameters.")
        best_model = DBSCAN(eps=1.5, min_samples=3)
        best_model.fit(X)
        best_params = {'eps': 1.5, 'min_samples': 3}
        best_score = -1
    
    print(f"\nBest DBSCAN: eps={best_params['eps']:.1f}, min_samples={best_params['min_samples']}")
    if best_score > 0:
        print(f"Silhouette score: {best_score:.3f}")
    
    return best_model, best_params


def cluster_optics(X, min_samples=3, xi=0.05, min_cluster_size=0.03):
    """OPTICS clustering: like DBSCAN but reveals multi-density structure.
    
    Produces a reachability plot where valleys indicate clusters.
    Better than DBSCAN for clusters of varying densities.
    
    Automatically extracts best clustering from multiple eps values.
    """
    print("\n=== OPTICS Clustering ===")
    n_samples = X.shape[0]
    min_cluster_size_abs = max(2, int(min_cluster_size * n_samples))
    
    # Use smaller min_samples for small datasets
    if n_samples < 15:
        min_samples = max(2, min_samples - 1)
    
    model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size_abs)
    _ = model.fit_predict(X)  # Fit to compute reachability
    
    # Try multiple eps cuts to find best clustering
    best_score = -1
    best_labels = model.labels_
    best_eps = None
    eps_cuts = [0.6, 0.8, 1.0]
    
    print(f"min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size_abs}")
    
    for eps_cut in eps_cuts:
        # Cluster at this eps level
        clusterer = OPTICS(min_samples=min_samples, max_eps=eps_cut, 
                          min_cluster_size=min_cluster_size_abs)
        labels = clusterer.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1:
            # Compute silhouette excluding noise
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > 1:
                score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                print(f"eps={eps_cut:.1f}: n_clusters={n_clusters}, noise={n_noise}, silhouette={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_eps = eps_cut
    
    if best_eps is not None:
        print(f"Best OPTICS eps: {best_eps:.1f}")
    else:
        print("Best OPTICS eps: auto (no valid clusters found)")
    
    if best_score > 0:
        print(f"Silhouette score: {best_score:.3f}")
    
    # Update model with best labels
    model.labels_ = best_labels
    
    return model, best_labels


def cluster_hierarchical(X, linkage='ward', distance_threshold=None):
    """Hierarchical Clustering (Ward linkage): bottom-up merges similar clusters.
    
    Ward minimizes the variance within clusters when merging.
    Dendrogram visualizes the merging process.
    """
    print("\n=== Hierarchical Clustering (Ward) ===")
    
    results = {}
    best_score = -1
    best_model = None
    best_k = None
    
    for k in [3, 4, 5]:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)
        
        score = silhouette_score(X, labels)
        results[k] = {'model': model, 'labels': labels, 'score': score}
        
        print(f"k={k}: silhouette={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_k = k
    
    print(f"\nBest Hierarchical: k={best_k}, silhouette={best_score:.3f}")
    
    return best_model, results, best_k


def compute_metrics(X, labels, method_name):
    """Calculate clustering quality metrics (excluding noise points for metrics)."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1) if -1 in labels else 0
    
    if n_clusters < 2:
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
    
    # Exclude noise points when computing metrics
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) <= 1:
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
    
    X_clean = X[non_noise_mask]
    labels_clean = labels[non_noise_mask]
    
    metrics = {
        'silhouette': silhouette_score(X_clean, labels_clean),
        'calinski_harabasz': calinski_harabasz_score(X_clean, labels_clean),
        'davies_bouldin': davies_bouldin_score(X_clean, labels_clean),
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }
    
    return metrics


def reduce_dimensions(X, n_components=2, use_umap=False):
    """Reduce features to 2D for visualization using PCA or UMAP."""
    if use_umap and UMAP_AVAILABLE:
        print("\nUsing UMAP for dimension reduction")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_2d = reducer.fit_transform(X)
        method_name = "UMAP"
    else:
        if use_umap:
            print("\nUMAP not available, falling back to PCA")
        else:
            print("\nUsing PCA for dimension reduction")
        pca = PCA(n_components=n_components, random_state=42)
        X_2d = pca.fit_transform(X)
        method_name = "PCA"
        print(f"Explained variance: {pca.explained_variance_ratio_[:2].sum():.3f}")
    
    return X_2d, method_name


def plot_clusters(X_2d, labels, title, method_name, outpath, player_names=None):
    """Plot 2D cluster visualization with player name labels."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(12, 10))
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(sorted(unique_labels), colors):
        if label == -1:
            marker = 'x'
            alpha = 0.6
            label_name = 'Noise'
            edgecolors = None
        else:
            marker = 'o'
            alpha = 0.8
            label_name = f'Cluster {label}'
            edgecolors = 'black'
        
        mask = labels == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=[color], marker=marker, s=100, 
                   alpha=alpha, label=label_name, edgecolors=edgecolors, linewidths=1)
        
        # Add player name labels
        if player_names is not None and label != -1:  # Only label clusters, not noise
            masked_indices = np.where(mask)[0]
            for idx, (x, y) in enumerate(X_2d[mask]):
                plt.annotate(player_names[masked_indices[idx]], (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    plt.title(f'{title} - {method_name}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{method_name} Component 1', fontsize=12)
    plt.ylabel(f'{method_name} Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def plot_optics_reachability(model, outpath):
    """Plot OPTICS reachability distances with horizontal guide lines."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(14, 7))
    
    reachability = model.reachability_[model.ordering_]
    
    plt.plot(reachability, linewidth=2, color='steelblue')
    
    # Add horizontal guide lines at multiple eps levels
    eps_levels = [0.6, 0.8, 1.0, 1.2]
    colors_list = ['green', 'orange', 'red', 'purple']
    for eps, color in zip(eps_levels, colors_list):
        if eps <= np.max(reachability):
            plt.axhline(y=eps, color=color, linestyle='--', alpha=0.5, linewidth=1.5, 
                       label=f'eps = {eps}')
    
    plt.title('OPTICS Reachability Plot (with eps thresholds)', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index (ordered)', fontsize=12)
    plt.ylabel('Reachability Distance', fontsize=12)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def plot_dendrogram(model_hier, X, k, outpath):
    """Plot hierarchical clustering dendrogram."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    Z = linkage(X, method='ward')
    
    dendrogram(Z, truncate_mode='lastp', p=k, leaf_rotation=90., 
               leaf_font_size=12., show_contracted=True)
    
    plt.title(f'Hierarchical Clustering Dendrogram (Ward Linkage)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


def save_cluster_summaries(df, labels_dict, X, feature_names, outdir):
    """Save per-cluster summaries with mean values and top distinctive features."""
    for method, labels in labels_dict.items():
        unique_labels = sorted(set(labels))
        
        # Skip noise-only clusters
        if len(unique_labels) == 1 and -1 in unique_labels:
            continue
        
        summaries = []
        
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # Skip noise
            
            # Get cluster members
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            
            if cluster_data.shape[0] == 0:
                continue
            
            # Compute mean values
            cluster_means = cluster_data.mean(axis=0)
            
            # Get overall means for comparison
            overall_means = X.mean(axis=0)
            
            # Find most distinctive features (largest deviation from overall mean)
            deviations = np.abs(cluster_means - overall_means)
            top_indices = np.argsort(deviations)[::-1][:3]
            top_features = [feature_names[i] for i in top_indices]
            
            # Create summary
            summary = {
                'method': method,
                'cluster_id': int(cluster_id),
                'n_players': int(cluster_mask.sum()),
                'top_feature_1': top_features[0] if len(top_features) > 0 else '',
                'top_feature_2': top_features[1] if len(top_features) > 1 else '',
                'top_feature_3': top_features[2] if len(top_features) > 2 else '',
            }
            
            # Add mean values for each feature
            for i, feat in enumerate(feature_names):
                summary[f'mean_{feat}'] = cluster_means[i]
            
            summaries.append(summary)
        
        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_path = os.path.join(outdir, f'cluster_summary_{method}.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved: {summary_path}")


def save_results(df, labels_dict, metrics_dict, outdir, X, feature_names):
    """Save labeled data, metrics, and cluster summaries to CSV and JSON."""
    os.makedirs(outdir, exist_ok=True)
    
    # Merge labels
    df_results = df.copy()
    for method, labels in labels_dict.items():
        df_results[f'cluster_{method}'] = labels
    
    # Save labeled players
    output_csv = os.path.join(outdir, 'player_labels.csv')
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved labeled players: {output_csv}")
    
    # Save metrics summary
    for method, metrics in metrics_dict.items():
        summary_df = pd.DataFrame([metrics])
        summary_path = os.path.join(outdir, f'summary_{method}.csv')
        summary_df.to_csv(summary_path, index=False)
    
    # Save all metrics as JSON
    metrics_json = json.dumps(metrics_dict, indent=2, default=str)
    with open(os.path.join(outdir, 'all_metrics.json'), 'w') as f:
        f.write(metrics_json)
    
    print(f"Saved metrics: {outdir}/all_metrics.json")
    
    # Save cluster summaries for each method
    save_cluster_summaries(df, labels_dict, X, feature_names, outdir)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Cluster tennis players by serve and rally metrics'
    )
    parser.add_argument('--input', type=str, default='data/players.csv',
                       help='Input CSV file path')
    parser.add_argument('--outdir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--use-umap', action='store_true',
                       help='Use UMAP for dimension reduction (default: PCA)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("="*70)
    print("TENNIS PLAYER CLUSTERING PROJECT")
    print("="*70)
    
    # Load data
    df = load_data(args.input)
    print(f"\nLoaded {len(df)} players")
    
    # Prepare features
    X, feature_names = prepare_features(df)
    
    # Reduce dimensions for visualization
    X_2d, viz_method = reduce_dimensions(X, use_umap=args.use_umap)
    
    # Clustering
    print("\n" + "="*70)
    print("CLUSTERING ANALYSIS")
    print("="*70)
    
    labels_dict = {}
    metrics_dict = {}
    
    # Extract player names for labeling
    player_names = df['player'].values
    
    # 1. DBSCAN
    model_dbscan, params_dbscan = cluster_dbscan(X)
    labels_dbscan = model_dbscan.labels_
    labels_dict['dbscan'] = labels_dbscan
    metrics_dict['dbscan'] = compute_metrics(X, labels_dbscan, 'DBSCAN')
    plot_clusters(X_2d, labels_dbscan, 'DBSCAN Clustering', viz_method,
                  os.path.join(args.outdir, f'{viz_method.lower()}_clusters_dbscan.png'),
                  player_names=player_names)
    
    # 2. OPTICS
    model_optics, labels_optics = cluster_optics(X)
    labels_dict['optics'] = labels_optics
    metrics_dict['optics'] = compute_metrics(X, labels_optics, 'OPTICS')
    plot_clusters(X_2d, labels_optics, 'OPTICS Clustering', viz_method,
                  os.path.join(args.outdir, f'{viz_method.lower()}_clusters_optics.png'),
                  player_names=player_names)
    plot_optics_reachability(model_optics,
                            os.path.join(args.outdir, 'optics_reachability.png'))
    
    # 3. Hierarchical
    model_hier, results_hier, best_k = cluster_hierarchical(X)
    labels_hier = results_hier[best_k]['labels']
    labels_dict['hierarchical'] = labels_hier
    metrics_dict['hierarchical'] = compute_metrics(X, labels_hier, 'Hierarchical')
    plot_clusters(X_2d, labels_hier, f'Hierarchical Clustering (k={best_k})', viz_method,
                  os.path.join(args.outdir, f'{viz_method.lower()}_clusters_hierarchical.png'),
                  player_names=player_names)
    plot_dendrogram(model_hier, X, best_k,
                   os.path.join(args.outdir, 'dendrogram_ward.png'))
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    save_results(df, labels_dict, metrics_dict, args.outdir, X, feature_names)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    summary_df = pd.DataFrame(metrics_dict).T
    summary_df = summary_df[['n_clusters', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'n_noise']]
    print("\nClustering Performance Metrics:")
    print(summary_df.to_string())
    
    # Best model
    valid_metrics = summary_df[summary_df['silhouette'].notna()]
    if len(valid_metrics) > 0:
        best_model = valid_metrics['silhouette'].idxmax()
        print(f"\n>>> Best clustering: {best_model.upper()}")
        print(f"  Silhouette score: {valid_metrics.loc[best_model, 'silhouette']:.3f}")
        print(f"  Number of clusters: {int(valid_metrics.loc[best_model, 'n_clusters'])}")
        
        # Interpretation
        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)
        print(f"The {best_model.upper()} method identified distinct playing styles.")
        print("Players in the same cluster share similar serve and rally characteristics.")
        if -1 in labels_dict[best_model]:
            noise_count = list(labels_dict[best_model]).count(-1)
            print(f"{noise_count} players were classified as outliers (noise).")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.outdir}/")


if __name__ == '__main__':
    main()

