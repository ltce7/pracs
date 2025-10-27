import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Generate different datasets
np.random.seed(42)

# Dataset 1: Moons
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

# Dataset 2: Blobs with outliers
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, n_features=2, 
                              cluster_std=0.6, random_state=42)
# Add outliers
outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X_blobs = np.vstack([X_blobs, outliers])
y_blobs = np.hstack([y_blobs, np.full(20, -1)])

# Dataset 3: Circles
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

datasets = [
    (X_moons, y_moons, "Two Moons"),
    (X_blobs, y_blobs, "Blobs with Outliers"),
    (X_circles, y_circles, "Two Circles")
]

print("DBSCAN Clustering Algorithm")
print("="*60)
print("DBSCAN = Density-Based Spatial Clustering of Applications with Noise")
print("Key Parameters:")
print("  - eps: Maximum distance between two samples")
print("  - min_samples: Minimum samples in a neighborhood for a core point")
print("="*60)

# Create main visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for idx, (X, y_true, name) in enumerate(datasets):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n{name} Dataset:")
    print("-" * 40)
    
    # Plot original data
    ax = axes[idx, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis',
                        edgecolors='black', alpha=0.7)
    ax.set_title(f'{name}\n(Original Data)')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Try different eps values
    eps_values = [0.3, 0.5]
    
    for eps_idx, eps in enumerate(eps_values):
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=5)
        y_pred = dbscan.fit_predict(X_scaled)
        
        # Count clusters and noise points
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        n_noise = list(y_pred).count(-1)
        
        print(f"\neps={eps}, min_samples=5:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        print(f"  Core samples: {len(dbscan.core_sample_indices_)}")
        
        # Calculate metrics (excluding noise points)
        if n_clusters > 1 and len(set(y_pred)) > 1:
            mask = y_pred != -1
            if sum(mask) > 0:
                silhouette = silhouette_score(X_scaled[mask], y_pred[mask])
                print(f"  Silhouette Score: {silhouette:.4f}")
        
        # Plot clustered data
        ax = axes[idx, eps_idx + 1]
        
        # Plot noise points
        noise_mask = y_pred == -1
        if sum(noise_mask) > 0:
            ax.scatter(X[noise_mask, 0], X[noise_mask, 1], 
                      c='red', marker='x', s=50, label='Noise', alpha=0.7)
        
        # Plot clustered points
        cluster_mask = y_pred != -1
        if sum(cluster_mask) > 0:
            scatter = ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1],
                               c=y_pred[cluster_mask], cmap='viridis',
                               edgecolors='black', alpha=0.7)
        
        ax.set_title(f'eps={eps}, min_samples=5\nClusters: {n_clusters}, Noise: {n_noise}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()

plt.tight_layout()
plt.show()

# Detailed example with parameter tuning
print("\n" + "="*60)
print("Detailed Parameter Analysis on Moons Dataset")
print("="*60)

X, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Try different combinations of eps and min_samples
eps_range = [0.2, 0.3, 0.4, 0.5]
min_samples_range = [3, 5, 10]

fig, axes = plt.subplots(len(min_samples_range), len(eps_range), figsize=(16, 12))

for i, min_samples in enumerate(min_samples_range):
    for j, eps in enumerate(eps_range):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        n_noise = list(y_pred).count(-1)
        
        ax = axes[i, j]
        
        # Plot noise points
        noise_mask = y_pred == -1
        if sum(noise_mask) > 0:
            ax.scatter(X[noise_mask, 0], X[noise_mask, 1],
                      c='red', marker='x', s=30, alpha=0.7)
        
        # Plot clustered points
        cluster_mask = y_pred != -1
        if sum(cluster_mask) > 0:
            ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1],
                      c=y_pred[cluster_mask], cmap='viridis',
                      edgecolors='black', alpha=0.7, s=30)
        
        ax.set_title(f'eps={eps}, min={min_samples}\nC:{n_clusters}, N:{n_noise}', 
                    fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('DBSCAN Parameter Grid Search', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Key Advantages of DBSCAN:")
print("="*60)
print("1. Can find arbitrarily shaped clusters")
print("2. Robust to outliers (marks them as noise)")
print("3. Does not require number of clusters as input")
print("4. Works well with varying density clusters")
print("\nKey Parameters to Tune:")
print("- eps: Smaller = more clusters, more noise")
print("- min_samples: Larger = fewer small clusters")