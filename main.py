#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tennis Player Clustering Project
Clusters professional tennis players by serve and rally metrics using DBSCAN, OPTICS, and Hierarchical clustering.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

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
from collections import defaultdict

# Optional UMAP
UMAP_AVAILABLE = False
umap: Optional[Any] = None
try:
    import umap as _umap  # type: ignore
    umap = _umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


# ------------------------ Data ------------------------

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


def load_data(input_path: str) -> pd.DataFrame:
    """Load tennis player data from CSV or generate synthetic data if missing."""
    if os.path.exists(input_path):
        print(f"Loading data from {input_path}")
        return pd.read_csv(input_path)
    else:
        print(f"File not found: {input_path}. Generating synthetic data...")
        df = generate_synthetic_data()
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        df.to_csv(input_path, index=False)
        print(f"Generated data saved to {input_path}")
        return df


def prepare_features(df: pd.DataFrame):
    """Extract and normalize numeric features for clustering."""
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


# ------------------------ Clustering ------------------------

def cluster_dbscan(X, eps_range=None, min_samples_range=None):
    """DBSCAN with small-grid search; metrics exclude noise."""
    if eps_range is None:
        eps_range = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]
    if min_samples_range is None:
        min_samples_range = [2, 3, 4, 5]

    best_score = -1.0
    best_model = None
    best_params = None

    print("\n=== DBSCAN Grid Search ===")
    for eps in eps_range:
        for min_samples in min_samples_range:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
            n_noise = int((labels == -1).sum())

            if n_clusters > 1:
                mask = labels != -1
                if int(mask.sum()) > 1:
                    score = silhouette_score(X[mask], labels[mask])
                    print(f"eps={eps:.1f}, min_samples={min_samples}: "
                          f"n_clusters={n_clusters}, noise={n_noise}, silhouette={score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = {'eps': eps, 'min_samples': min_samples}

    if best_model is None:
        print("No valid clusters found with grid search. Using fallback parameters.")
        best_model = DBSCAN(eps=1.5, min_samples=3).fit(X)
        best_params = {'eps': 1.5, 'min_samples': 3}
    else:
        best_model.fit(X)

    print(f"\nBest DBSCAN: eps={best_params['eps']:.1f}, min_samples={best_params['min_samples']}")
    if best_score > 0:
        print(f"Silhouette score: {best_score:.3f}")
    return best_model, best_params


def cluster_optics(X, min_samples=3, xi=0.05, min_cluster_size=0.03):
    """OPTICS with reachability and multiple eps 'cuts' for small datasets."""
    print("\n=== OPTICS Clustering ===")
    n_samples = X.shape[0]
    min_cluster_size_abs = max(2, int(min_cluster_size * n_samples))
    if n_samples < 15:
        min_samples = max(2, min_samples - 1)

    base = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size_abs)
    _ = base.fit_predict(X)  # fit to compute reachability

    # try several eps values and pick best silhouette
    best_score = -1.0
    best_labels = base.labels_.copy()
    best_eps = None
    for eps_cut in [0.6, 0.8, 1.0]:
        clusterer = OPTICS(min_samples=min_samples, max_eps=eps_cut,
                           min_cluster_size=min_cluster_size_abs)
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
        n_noise = int((labels == -1).sum())
        if n_clusters > 1:
            mask = labels != -1
            if int(mask.sum()) > 1:
                score = silhouette_score(X[mask], labels[mask])
                print(f"eps={eps_cut:.1f}: n_clusters={n_clusters}, noise={n_noise}, silhouette={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_labels = labels.copy()
                    best_eps = eps_cut

    if best_eps is not None:
        print(f"Best OPTICS eps: {best_eps:.1f}")
    else:
        print("Best OPTICS eps: auto (no valid clusters found)")

    if best_score > 0:
        print(f"Silhouette score: {best_score:.3f}")

    # attach best labels to model we return
    base.labels_ = best_labels
    return base, best_labels


def cluster_hierarchical(X, linkage='ward'):
    """Agglomerative clustering with Ward linkage; pick best k by silhouette."""
    print("\n=== Hierarchical Clustering (Ward) ===")
    results = {}
    best_score = -1.0
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
    """Quality metrics; exclude noise for DBSCAN/OPTICS."""
    n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
    n_noise = int((labels == -1).sum()) if -1 in set(labels) else 0
    if n_clusters < 2:
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
    mask = labels != -1
    if int(mask.sum()) <= 1:
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
    Xc = X[mask]
    Lc = labels[mask]
    return {
        'silhouette': silhouette_score(Xc, Lc),
        'calinski_harabasz': calinski_harabasz_score(Xc, Lc),
        'davies_bouldin': davies_bouldin_score(Xc, Lc),
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }


# ------------------------ Dimensionality reduction ------------------------

def reduce_dimensions(X, n_components=2, use_umap=False):
    """Reduce to 2D with PCA (default) or UMAP. Returns (X_2d, method_name, fitted_model)."""
    if use_umap and UMAP_AVAILABLE and umap is not None:
        print("\nUsing UMAP for dimension reduction")
        reducer = umap.UMAP(n_components=n_components, random_state=42)  # type: ignore
        X_2d = reducer.fit_transform(X)
        method_name = "UMAP"
        fitted = reducer
    else:
        if use_umap and not UMAP_AVAILABLE:
            print("\nUMAP not available, falling back to PCA")
        else:
            print("\nUsing PCA for dimension reduction")
        pca = PCA(n_components=n_components, random_state=42)
        X_2d = pca.fit_transform(X)
        method_name = "PCA"
        print(f"Explained variance: {pca.explained_variance_ratio_[:2].sum():.3f}")
        fitted = pca
    return X_2d, method_name, fitted


# ------------------------ Interpretability helpers ------------------------

def pca_axis_labels(pca: PCA, feature_names, top_k=2):
    """Build human-readable axis labels from PCA loadings."""
    def label_for_pc(pc_idx: int) -> str:
        load = pca.components_[pc_idx]  # (n_features,)
        top_idx = np.argsort(np.abs(load))[::-1][:top_k]
        top_feats = [feature_names[i] for i in top_idx]
        pretty = {
            'serve_speed_avg': 'Serve Speed',
            'ace_pct': 'Ace %',
            'double_fault_pct': 'Double Fault %',
            'first_serve_in_pct': '1st Serve In %',
            'first_serve_points_won_pct': '1st Serve Pts Won %',
            'break_points_saved_pct': 'Break Pts Saved %',
            'avg_rally_length': 'Avg Rally Length',
            'win_ratio': 'Win Ratio',
            'height_cm': 'Height (cm)',
        }
        pretty_feats = [pretty.get(f, f) for f in top_feats]

        # heuristic naming
        if 'serve_speed_avg' in top_feats or 'ace_pct' in top_feats:
            base = "Serve Aggressiveness"
        elif 'avg_rally_length' in top_feats:
            base = "Rally Endurance"
        elif 'first_serve_points_won_pct' in top_feats or 'break_points_saved_pct' in top_feats:
            base = "Point Conversion/Defense"
        else:
            base = "Mixed Traits"
        return f"{base} (PCA {pc_idx+1}: {', '.join(pretty_feats)})"

    return label_for_pc(0), label_for_pc(1)


def infer_cluster_name_from_means(mean_row: dict) -> str:
    """
    Returns a base style name from ORIGINAL-scale cluster means.
    Rules:
      - High serve speed / ace, shorter rallies -> 'Power Servers'
      - Long rallies, moderate serve -> 'Baseliners'
      - Else -> 'All-Rounders'
    """
    ss = float(mean_row.get('serve_speed_avg', 0))
    ace = float(mean_row.get('ace_pct', 0))
    rally = float(mean_row.get('avg_rally_length', 0))
    fs_won = float(mean_row.get('first_serve_points_won_pct', 0))

    if (ss >= 200 or ace >= 0.14) and rally <= 10.0:
        return "Power Servers"
    if rally >= 11.0 and ss <= 195:
        return "Baseliners"
    if fs_won >= 0.77 and 9.5 <= rally <= 11.0:
        return "All-Rounders"
    return "All-Rounders"


def _subtype_token(base: str, mean_row: dict, overall_means: dict) -> str:
    """
    Create a short, human-readable subtype tag to distinguish clusters
    that share the same base style.
    """
    def hi_lo(val, ref, hi, lo):
        return hi if val >= ref else lo

    if base == "Power Servers":
        # Split by height (tall vs compact) or by ace%
        token = hi_lo(
            float(mean_row.get('height_cm', 0)),
            float(overall_means.get('height_cm', 0)),
            "Tall", "Compact"
        )
        # add a 2nd hint on rally length
        rally = float(mean_row.get('avg_rally_length', 0))
        token2 = "Short Rally" if rally <= float(overall_means.get('avg_rally_length', 0)) else "Longer Rally"
        return f"{token}, {token2}"

    if base == "Baseliners":
        # Emphasize rally length and break defense
        token = hi_lo(
            float(mean_row.get('avg_rally_length', 0)),
            float(overall_means.get('avg_rally_length', 0)),
            "Long Rally", "Medium Rally"
        )
        token2 = hi_lo(
            float(mean_row.get('break_points_saved_pct', 0)),
            float(overall_means.get('break_points_saved_pct', 0)),
            "Strong Defense", "Balanced Defense"
        )
        return f"{token}, {token2}"

    # All-Rounders: emphasize 1st-serve points won and win ratio
    token = hi_lo(
        float(mean_row.get('first_serve_points_won_pct', 0)),
        float(overall_means.get('first_serve_points_won_pct', 0)),
        "High 1st-Serve Pts", "Balanced 1st-Serve Pts"
    )
    token2 = hi_lo(
        float(mean_row.get('win_ratio', 0)),
        float(overall_means.get('win_ratio', 0)),
        "High WR", "Balanced WR"
    )
    return f"{token}, {token2}"


def build_cluster_name_map(df_original_numeric, labels, feature_names):
    """
    Returns a dict {cluster_label: unique_human_name} (ignores noise -1).
    Names are built as: '<Base Style> — <Subtype>', and uniqueness is enforced
    with #2, #3… suffixes if needed.
    """
    import pandas as pd
    out = {}
    valid_clusters = sorted(set(labels) - {-1})
    if not valid_clusters:
        return out

    # Work in ORIGINAL feature space
    tmp = pd.DataFrame(df_original_numeric, columns=feature_names)
    tmp['__lab__'] = labels

    # overall means used for subtype tokens
    overall_means = tmp[feature_names].mean().to_dict()

    # track duplicates
    seen = defaultdict(int)

    for c in valid_clusters:
        mean_row = tmp[tmp['__lab__'] == c][feature_names].mean().to_dict()
        base = infer_cluster_name_from_means(mean_row)
        subtype = _subtype_token(base, mean_row, overall_means)
        name = f"{base} — {subtype}"

        seen[name] += 1
        if seen[name] > 1:
            # If another cluster got the exact same title, make it unique
            name = f"{name} #{seen[name]}"

        out[c] = name

    return out


# ------------------------ Plotting ------------------------

def plot_clusters(X_2d, labels, title, method_name, outpath, names=None, axis_labels=None, cluster_name_map=None):
    """2D cluster plot with semantic axis labels and human cluster names."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig = plt.figure(figsize=(10, 8))

    unique_labels = sorted(set(map(int, np.unique(labels))))
    cmap = plt.cm.get_cmap('Spectral', max(len(unique_labels), 1))

    for idx, label in enumerate(unique_labels):
        color = cmap(idx)
        mask = (labels == label)
        if label == -1:
            marker = 'x'
            alpha = 0.7
            leg = 'Noise'
            edgecolors = None
        else:
            marker = 'o'
            alpha = 0.85
            human = (cluster_name_map or {}).get(label, f'Cluster {label}')
            leg = human
            edgecolors = 'black'

        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=[color], marker=marker, s=110, alpha=alpha,
                    label=leg, edgecolors=edgecolors, linewidths=0.7)

    if names is not None:
        for i, n in enumerate(names):
            plt.text(float(X_2d[i, 0]), float(X_2d[i, 1]), str(n), fontsize=8, alpha=0.7)

    if axis_labels and len(axis_labels) == 2:
        plt.xlabel(axis_labels[0], fontsize=12)
        plt.ylabel(axis_labels[1], fontsize=12)
    else:
        plt.xlabel(f'{method_name} Component 1', fontsize=12)
        plt.ylabel(f'{method_name} Component 2', fontsize=12)

    plt.title(f'{title} - {method_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_optics_reachability(model: OPTICS, outpath: str):
    """OPTICS reachability plot with horizontal eps guides."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig = plt.figure(figsize=(14, 7))

    reachability = model.reachability_[model.ordering_]
    plt.plot(reachability, linewidth=2)

    eps_levels = [0.6, 0.8, 1.0, 1.2]
    for eps in eps_levels:
        if np.isfinite(reachability).any() and eps <= float(np.nanmax(reachability)):
            plt.axhline(y=eps, linestyle='--', alpha=0.5, linewidth=1.5, label=f'eps = {eps}')

    plt.title('OPTICS Reachability Plot (with eps thresholds)', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index (ordered)', fontsize=12)
    plt.ylabel('Reachability Distance', fontsize=12)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_dendrogram(model_hier, X, k, outpath):
    """Hierarchical clustering dendrogram (Ward linkage)."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig = plt.figure(figsize=(12, 8))

    Z = linkage(X, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=k, leaf_rotation=90.,
               leaf_font_size=12., show_contracted=True)

    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


# ------------------------ Saving ------------------------

def save_cluster_summaries(df: pd.DataFrame, labels_dict: dict, X: np.ndarray, feature_names, outdir: str):
    """Per-cluster summaries (on SCALED features: mean + top deviations)."""
    os.makedirs(outdir, exist_ok=True)
    for method, labels in labels_dict.items():
        unique_labels = sorted(set(map(int, np.unique(labels))))
        if unique_labels == [-1]:
            continue

        rows = []
        for cid in unique_labels:
            if cid == -1:
                continue
            mask = (labels == cid)
            if int(mask.sum()) == 0:
                continue
            cl_means = X[mask].mean(axis=0)
            overall = X.mean(axis=0)
            dev = np.abs(cl_means - overall)
            top_idx = np.argsort(dev)[::-1][:3]
            row = {
                'method': method,
                'cluster_id': int(cid),
                'n_players': int(mask.sum()),
                'top_feature_1': feature_names[top_idx[0]] if len(top_idx) > 0 else '',
                'top_feature_2': feature_names[top_idx[1]] if len(top_idx) > 1 else '',
                'top_feature_3': feature_names[top_idx[2]] if len(top_idx) > 2 else '',
            }
            for i, f in enumerate(feature_names):
                row[f'mean_{f}'] = float(cl_means[i])
            rows.append(row)

        if rows:
            out_df = pd.DataFrame(rows)
            path = os.path.join(outdir, f'cluster_summary_{method}.csv')
            out_df.to_csv(path, index=False)
            print(f"Saved: {path}")


def save_results(df: pd.DataFrame, labels_dict: dict, metrics_dict: dict, outdir: str, X: np.ndarray, feature_names):
    """Save labels, metrics, and summaries."""
    os.makedirs(outdir, exist_ok=True)
    df_out = df.copy()
    for method, labels in labels_dict.items():
        df_out[f'cluster_{method}'] = labels
    labels_csv = os.path.join(outdir, 'player_labels.csv')
    df_out.to_csv(labels_csv, index=False)
    print(f"\nSaved labeled players: {labels_csv}")

    # per-method metric CSV and combined JSON
    for method, metrics in metrics_dict.items():
        mdf = pd.DataFrame([metrics])
        mdf.to_csv(os.path.join(outdir, f'summary_{method}.csv'), index=False)
    with open(os.path.join(outdir, 'all_metrics.json'), 'w') as f:
        f.write(json.dumps(metrics_dict, indent=2, default=str))
    print(f"Saved metrics: {outdir}/all_metrics.json")

    save_cluster_summaries(df, labels_dict, X, feature_names, outdir)


# ------------------------ Main ------------------------

def main():
    parser = argparse.ArgumentParser(description='Cluster tennis players by serve and rally metrics')
    parser.add_argument('--input', type=str, default='data/players.csv', help='Input CSV file path')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory for results')
    parser.add_argument('--use-umap', action='store_true', help='Use UMAP for 2D viz (default: PCA)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("TENNIS PLAYER CLUSTERING PROJECT")
    print("=" * 70)

    # Load & features
    df = load_data(args.input)
    print(f"\nLoaded {len(df)} players")

    X, feature_names = prepare_features(df)

    # 2D projection
    X_2d, viz_method, reducer = reduce_dimensions(X, use_umap=args.use_umap)

    # PCA axis labels (if PCA)
    axis_labels = None
    if viz_method == "PCA" and isinstance(reducer, PCA):
        axis_labels = pca_axis_labels(reducer, feature_names, top_k=2)

    # original-scale numeric for naming clusters
    df_numeric_original = df[feature_names].copy().values
    player_names = df['player'].values if 'player' in df.columns else None

    # -------- Clustering --------
    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS")
    print("=" * 70)

    labels_dict = {}
    metrics_dict = {}

    # DBSCAN
    model_dbscan, params_dbscan = cluster_dbscan(X)
    labels_dbscan = model_dbscan.labels_
    labels_dict['dbscan'] = labels_dbscan
    metrics_dict['dbscan'] = compute_metrics(X, labels_dbscan, 'DBSCAN')

    name_map_db = build_cluster_name_map(df_numeric_original, labels_dbscan, feature_names)
    plot_clusters(
        X_2d, labels_dbscan, 'DBSCAN Clustering', viz_method,
        os.path.join(args.outdir, f'{viz_method.lower()}_clusters_dbscan.png'),
        names=player_names, axis_labels=axis_labels, cluster_name_map=name_map_db
    )

    # OPTICS
    model_optics, labels_optics = cluster_optics(X)
    labels_dict['optics'] = labels_optics
    metrics_dict['optics'] = compute_metrics(X, labels_optics, 'OPTICS')

    name_map_opt = build_cluster_name_map(df_numeric_original, labels_optics, feature_names)
    plot_clusters(
        X_2d, labels_optics, 'OPTICS Clustering', viz_method,
        os.path.join(args.outdir, f'{viz_method.lower()}_clusters_optics.png'),
        names=player_names, axis_labels=axis_labels, cluster_name_map=name_map_opt
    )
    plot_optics_reachability(
        model_optics, os.path.join(args.outdir, 'optics_reachability.png')
    )

    # Hierarchical
    model_hier, results_hier, best_k = cluster_hierarchical(X)
    labels_hier = results_hier[best_k]['labels']
    labels_dict['hierarchical'] = labels_hier
    metrics_dict['hierarchical'] = compute_metrics(X, labels_hier, 'Hierarchical')

    name_map_hier = build_cluster_name_map(df_numeric_original, labels_hier, feature_names)
    plot_clusters(
        X_2d, labels_hier, f'Hierarchical Clustering (k={best_k})', viz_method,
        os.path.join(args.outdir, f'{viz_method.lower()}_clusters_hierarchical.png'),
        names=player_names, axis_labels=axis_labels, cluster_name_map=name_map_hier
    )
    plot_dendrogram(
        model_hier, X, best_k, os.path.join(args.outdir, 'dendrogram_ward.png')
    )

    # -------- Save results --------
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    save_results(df, labels_dict, metrics_dict, args.outdir, X, feature_names)

    # save human-readable name maps
    for meth, labels in labels_dict.items():
        cmap = build_cluster_name_map(df_numeric_original, labels, feature_names)
        if cmap:
            pd.DataFrame(
                [{'cluster_label': k, 'cluster_name': v} for k, v in cmap.items()]
            ).to_csv(os.path.join(args.outdir, f'cluster_names_{meth}.csv'), index=False)

    # -------- Summary --------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(metrics_dict).T
    summary_df = summary_df[['n_clusters', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'n_noise']]
    print("\nClustering Performance Metrics:")
    print(summary_df.to_string())

    valid_metrics = summary_df[summary_df['silhouette'].notna()]
    if len(valid_metrics) > 0:
        best_model_key = valid_metrics['silhouette'].astype(float).idxmax()
        print(f"\n>>> Best clustering: {str(best_model_key).upper()}")
        print(f"  Silhouette score: {float(valid_metrics.loc[best_model_key, 'silhouette']):.3f}")
        print(f"  Number of clusters: {int(valid_metrics.loc[best_model_key, 'n_clusters'])}")

        print("\n" + "-" * 70)
        print("INTERPRETATION")
        print("-" * 70)
        print(f"The {str(best_model_key).upper()} method identified distinct playing styles.")
        print("Players in the same cluster share similar serve and rally characteristics.")
        lab_arr = np.asarray(labels_dict[best_model_key])
        if (-1 in lab_arr):
            noise_count = int((lab_arr == -1).sum())
            print(f"{noise_count} players were classified as outliers (noise).")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.outdir}/")


if __name__ == '__main__':
    main()


