import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Generate different types of datasets
np.random.seed(42)

# Dataset 1: Two moons (non-linearly separable)
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

# Dataset 2: Two circles (non-linearly separable)
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# Dataset 3: Blobs (linearly separable)
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, n_features=2, 
                              cluster_std=0.6, random_state=42)

datasets = [
    (X_moons, y_moons, "Two Moons", 2),
    (X_circles, y_circles, "Two Circles", 2),
    (X_blobs, y_blobs, "Blobs", 3)
]

print("Graph-Based Clustering (Spectral Clustering)")
print("="*60)

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for idx, (X, y_true, name, n_clusters) in enumerate(datasets):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n{name} Dataset:")
    print("-" * 40)
    
    # Plot 1: Original data with true labels
    ax = axes[idx, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
                        edgecolors='black', alpha=0.7)
    ax.set_title(f'{name}\n(True Labels)')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax)
    
    # Apply Spectral Clustering with different affinity measures
    affinities = ['nearest_neighbors', 'rbf']
    
    for aff_idx, affinity in enumerate(affinities):
        # Apply Spectral Clustering
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=42,
            assign_labels='kmeans',
            n_neighbors=10
        )
        
        y_pred = spectral.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        
        print(f"\nAffinity: {affinity}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Adjusted Rand Index: {ari:.4f}")
        
        # Plot clustered data
        ax = axes[idx, aff_idx + 1]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis',
                           edgecolors='black', alpha=0.7)
        ax.set_title(f'{affinity.upper()}\nSilhouette: {silhouette:.3f}, ARI: {ari:.3f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

# Additional example: Spectral Clustering on a complex dataset
print("\n" + "="*60)
print("Detailed Example: Spectral Clustering on Moons Dataset")
print("="*60)

X, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Try different number of neighbors
n_neighbors_list = [5, 10, 20]

fig, axes = plt.subplots(1, len(n_neighbors_list), figsize=(15, 5))

for i, n_neighbors in enumerate(n_neighbors_list):
    spectral = SpectralClustering(
        n_clusters=2,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=42
    )
    y_pred = spectral.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"\nn_neighbors = {n_neighbors}:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis',
                   edgecolors='black', alpha=0.7)
    axes[i].set_title(f'n_neighbors={n_neighbors}\nSil: {silhouette:.3f}, ARI: {ari:.3f}')
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Key Points about Spectral Clustering:")
print("="*60)
print("1. Works well with non-linearly separable data")
print("2. Uses graph-based similarity (affinity matrix)")
print("3. Performs eigenvalue decomposition")
print("4. Better than K-means for complex cluster shapes")