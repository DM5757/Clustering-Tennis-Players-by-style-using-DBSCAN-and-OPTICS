# Tennis Player Clustering Project

Cluster professional tennis players by serve and rally metrics using three unsupervised learning algorithms: **DBSCAN**, **OPTICS**, and **Hierarchical Clustering**.

## Overview

This project analyzes tennis player performance data to identify distinct playing styles such as power servers, baseliners, and all-rounders. The analysis uses multiple clustering algorithms to group players based on metrics like serve speed, ace percentage, rally length, and win ratio.

## Algorithms

### DBSCAN
Density-Based Spatial Clustering of Applications with Noise. DBSCAN groups points that are closely packed together, marking low-density regions as outliers. It can discover arbitrarily shaped clusters and doesn't require specifying the number of clusters beforehand. Parameters: `eps` (neighborhood radius) and `min_samples` (minimum points to form a dense region).

### OPTICS
Ordering Points To Identify the Clustering Structure. Similar to DBSCAN but better suited for datasets with varying densities. OPTICS produces a reachability plot where valleys indicate clusters, providing insight into the cluster structure across different density levels.

### Hierarchical Clustering (Ward Linkage)
An agglomerative bottom-up approach that starts with individual points and merges similar clusters iteratively. Ward linkage minimizes the variance within clusters during merging. The dendrogram visualizes the hierarchical merging process, making it easy to identify the optimal number of clusters by cutting the tree at different levels.

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic execution
```bash
python main.py --input data/players.csv --outdir outputs
```

### Command-line options
- `--input`: Path to input CSV file (default: `data/players.csv`)
- `--outdir`: Output directory for results (default: `outputs`)
- `--use-umap`: Use UMAP for 2D visualization instead of PCA
- `--seed`: Random seed for reproducibility (default: 42)

### Examples
```bash
# Use UMAP for dimension reduction
python main.py --use-umap

# Specify custom input and output directories
python main.py --input my_data.csv --outdir results

# Set random seed
python main.py --seed 123
```

## Output Files

The script generates the following outputs in the specified output directory:

### CSV Files
- `player_labels.csv`: Original data with cluster labels from each algorithm
- `summary_dbscan.csv`: DBSCAN performance metrics
- `summary_optics.csv`: OPTICS performance metrics
- `summary_hierarchical.csv`: Hierarchical clustering performance metrics

### Visualizations (PNG)
- `pca_clusters_dbscan.png`: DBSCAN clusters in 2D space
- `pca_clusters_optics.png`: OPTICS clusters in 2D space
- `pca_clusters_hierarchical.png`: Hierarchical clusters in 2D space
- `optics_reachability.png`: OPTICS reachability plot
- `dendrogram_ward.png`: Hierarchical clustering dendrogram

### Metrics
- `all_metrics.json`: Comprehensive clustering metrics in JSON format

## Performance Metrics

Each clustering method is evaluated using:
- **Silhouette Score**: Measures how similar objects are to their own cluster vs others (-1 to 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance (higher is better)
- **Davies-Bouldin Index**: Average similarity between clusters (lower is better)

The script automatically selects the best-performing model based on the Silhouette Score.

## Data Format

Input CSV should contain the following columns:
- `player`: Player name
- `handedness`: Right/Left/Both
- `dominant_surface`: Hard/Clay/Grass
- `serve_speed_avg`: Average serve speed (km/h)
- `ace_pct`: Ace percentage
- `double_fault_pct`: Double fault percentage
- `first_serve_in_pct`: First serve in percentage
- `first_serve_points_won_pct`: First serve win percentage
- `break_points_saved_pct`: Break points saved percentage
- `avg_rally_length`: Average rally length (shots)
- `win_ratio`: Overall win ratio
- `height_cm`: Height in centimeters

If the input file is missing, the script will automatically generate synthetic data with varied playing styles.

## Project Structure

```
.
├── main.py              # Main clustering script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/
│   └── players.csv     # Input data (auto-generated if missing)
└── outputs/            # Generated results
    ├── player_labels.csv
    ├── all_metrics.json
    ├── pca_clusters_*.png
    ├── optics_reachability.png
    └── dendrogram_ward.png
```

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- scipy
- umap-learn (optional, for UMAP visualization)

## License

MIT License

## Author

Python Engineer & Data Scientist

