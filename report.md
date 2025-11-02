🎾 Clustering Professional Tennis Players by Playing Style Using DBSCAN, OPTICS, and Hierarchical Algorithms
Author: Davit Machitidze
Course: Numerical Programming
Abstract

This project applies unsupervised machine learning to identify distinct playing styles among professional tennis players using real and synthetic performance data. Three clustering algorithms — DBSCAN, OPTICS, and Hierarchical Clustering (Ward linkage) — are used to detect natural groupings of players based on metrics describing serve power, rally behavior, and consistency.
The analysis visualizes results using PCA (Principal Component Analysis) for dimensionality reduction, revealing clusters that correspond to well-known tennis archetypes such as Power Servers and Baseliners. Each algorithm’s performance is compared using internal validation metrics, and the final results demonstrate how unsupervised learning can capture meaningful structure in sports performance data.

1. Introduction

Professional tennis players vary greatly in how they construct points — some rely on fast serves and short rallies, while others depend on defense and consistency.
The goal of this project is to automatically group players by style based solely on match statistics, without predefined labels.

By applying clustering techniques, we can uncover hidden structure in player data, identify statistically similar athletes, and compare the strengths of different clustering algorithms.
This approach can be extended to other domains in sports analytics where player behavior needs to be discovered rather than classified.

2. Data and Features

The dataset consists of 15 professional tennis players, represented by nine quantitative attributes that describe their serve, rally, and performance profiles:

Category	Features
Serve	serve_speed_avg, ace_pct, double_fault_pct, first_serve_in_pct, first_serve_points_won_pct
Defense	break_points_saved_pct
Rally	avg_rally_length
Performance	win_ratio
Physical	height_cm

If no CSV file is provided, the script generates synthetic data with realistic distributions for three general playing tendencies. All numeric features are standardized (z-score normalization) before clustering to ensure each metric contributes equally.

3. Methodology
3.1 Dimensionality Reduction

The high-dimensional data are projected to two principal components using PCA for easier visualization:

PCA 1 (x-axis): Rally Endurance — combines rally length and serve consistency.

PCA 2 (y-axis): Serve Aggressiveness — reflects serve speed, ace rate, and height.

These axes allow intuitive plots of player similarity in 2D space.

3.2 Clustering Algorithms
DBSCAN

A density-based algorithm that groups points closely packed together and labels sparse points as noise.
It identifies clusters of arbitrary shape without pre-defining the number of clusters.
Parameters: eps (radius of neighborhood) and min_samples (minimum number of points to form a dense area).

OPTICS

An extension of DBSCAN that handles varying density.
It produces a reachability plot where valleys correspond to clusters and peaks indicate gaps or outliers.
This plot visually represents cluster structure at different density levels.

Hierarchical Clustering (Ward Linkage)

A bottom-up method that begins with each player as its own cluster and merges the most similar ones iteratively.
Ward linkage minimizes within-cluster variance. The number of clusters (k = 3 here) is chosen based on the silhouette score.

4. Results and Interpretation
4.1 Detected Playing Styles

All three algorithms consistently revealed two main stylistic groups plus occasional outliers:

Cluster	Description	Example Players
Power Servers — Tall, Short Rally	High serve speed, many aces, short rallies, tall height	Marin Cilic, Taylor Fritz, John Isner
Power Servers — Compact, Short Rally	Slightly shorter but equally serve-dominant players	Jannik Sinner, Andrey Rublev, Roger Federer
Baseliners — Long Rally, Balanced Defense	Longer rallies, strong return games, high consistency	Rafael Nadal, Novak Djokovic, Stefanos Tsitsipas, Carlos Alcaraz

Cluster names are derived automatically from the average values of serve and rally metrics in each group.

4.2 Outliers and Noise

In DBSCAN, several elite players — notably Novak Djokovic, Rafael Nadal, and Carlos Alcaraz — were marked as noise.
This is because their balanced statistical profiles lie between the two density clusters rather than within one.
For example, Djokovic combines moderate serve power with extremely high defensive ability and rally endurance, which makes him unique in density terms.

Hierarchical clustering, which relies on overall distance rather than density, successfully grouped these players under the Baseliner style, confirming its robustness for small datasets.

4.3 OPTICS Reachability Plot

The OPTICS reachability diagram plots each player’s reachability distance (y-axis) against their order in the data (x-axis).

Deep valleys correspond to dense player groups — the clusters.

High peaks represent players statistically distant from any cluster — the outliers.
Horizontal dashed lines (eps = 0.6 – 1.2) show possible thresholds for cluster separation.
This confirms that the dataset supports two to three stylistic clusters of varying density.

5. Evaluation Metrics

To compare performance, three internal metrics were used:

Silhouette Score: Higher values indicate better separation between clusters.

Calinski–Harabasz Index: Ratio of between- to within-cluster dispersion (higher = better).

Davies–Bouldin Index: Measures cluster overlap (lower = better).

The Hierarchical (Ward) model with k = 3 achieved the highest silhouette score, indicating the clearest partition between Power Servers and Baseliners.

6. Conclusion

This project demonstrates how unsupervised learning can identify distinct tennis play styles from statistical data alone.
The algorithms rediscovered recognizable categories such as Power Servers and Baseliners, and successfully visualized these in two-dimensional PCA space.
Differences between algorithms highlight how density-based methods (DBSCAN, OPTICS) emphasize local similarity, while hierarchical methods reveal broader stylistic structure.

Overall, the analysis confirms that clustering is a powerful exploratory tool for sports analytics, capable of summarizing complex multi-dimensional performance data into meaningful human-interpretable groups.

References

Ester et al. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN).

Ankerst et al. (1999). OPTICS: Ordering Points To Identify the Clustering Structure.

Ward, J.H. (1963). Hierarchical Grouping to Optimize an Objective Function.

Scikit-Learn Documentation. Cluster Evaluation Metrics.