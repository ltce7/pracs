import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

print("="*70)
print("DIMENSIONALITY REDUCTION TECHNIQUES")
print("="*70)
print("1. PCA - Principal Component Analysis (Unsupervised)")
print("2. SVD - Singular Value Decomposition (Unsupervised)")
print("3. LDA - Linear Discriminant Analysis (Supervised)")
print("="*70)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Standardize the features
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

print(f"\nOriginal Iris dataset shape: {X_iris.shape}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")

# ==================== PCA ====================
print("\n" + "="*70)
print("1. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*70)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris_scaled)

print(f"\nReduced shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
print(f"\nPrincipal Components:")
print(pca.components_)

# Variance explained by each component
print("\nVariance explained by each component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")

# Analyze all components
pca_full = PCA()
pca_full.fit(X_iris_scaled)

# ==================== SVD ====================
print("\n" + "="*70)
print("2. SINGULAR VALUE DECOMPOSITION (SVD)")
print("="*70)

# Apply SVD (TruncatedSVD for dimensionality reduction)
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_iris_scaled)

print(f"\nReduced shape: {X_svd.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_}")
print(f"Total variance explained: {sum(svd.explained_variance_ratio_):.4f}")
print(f"\nSingular values: {svd.singular_values_}")

# ==================== LDA ====================
print("\n" + "="*70)
print("3. LINEAR DISCRIMINANT ANALYSIS (LDA)")
print("="*70)

# Apply LDA (supervised, uses class labels)
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_iris_scaled, y_iris)

print(f"\nReduced shape: {X_lda.shape}")
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
print(f"Total variance explained: {sum(lda.explained_variance_ratio_):.4f}")

# ==================== VISUALIZATION ====================
fig = plt.figure(figsize=(18, 12))

# Plot 1: PCA
ax1 = plt.subplot(2, 3, 1)
for target, color in zip([0, 1, 2], ['red', 'green', 'blue']):
    mask = y_iris == target
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=color, label=iris.target_names[target], alpha=0.7, edgecolors='black')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
ax1.set_title('PCA - Iris Dataset')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: SVD
ax2 = plt.subplot(2, 3, 2)
for target, color in zip([0, 1, 2], ['red', 'green', 'blue']):
    mask = y_iris == target
    ax2.scatter(X_svd[mask, 0], X_svd[mask, 1],
               c=color, label=iris.target_names[target], alpha=0.7, edgecolors='black')
ax2.set_xlabel('First Component')
ax2.set_ylabel('Second Component')
ax2.set_title('SVD - Iris Dataset')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: LDA
ax3 = plt.subplot(2, 3, 3)
for target, color in zip([0, 1, 2], ['red', 'green', 'blue']):
    mask = y_iris == target
    ax3.scatter(X_lda[mask, 0], X_lda[mask, 1],
               c=color, label=iris.target_names[target], alpha=0.7, edgecolors='black')
ax3.set_xlabel('LD1')
ax3.set_ylabel('LD2')
ax3.set_title('LDA - Iris Dataset')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: PCA Explained Variance
ax4 = plt.subplot(2, 3, 4)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
ax4.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=8)
ax4.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
ax4.set_xlabel('Number of Components')
ax4.set_ylabel('Cumulative Explained Variance')
ax4.set_title('PCA - Cumulative Explained Variance')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Feature importance in PCA
ax5 = plt.subplot(2, 3, 5)
components = pca.components_
feature_names = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W']
x = np.arange(len(feature_names))
width = 0.35
ax5.bar(x - width/2, components[0], width, label='PC1', alpha=0.8)
ax5.bar(x + width/2, components[1], width, label='PC2', alpha=0.8)
ax5.set_xlabel('Features')
ax5.set_ylabel('Component Loading')
ax5.set_title('PCA - Feature Loadings')
ax5.set_xticks(x)
ax5.set_xticklabels(feature_names, rotation=45)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Comparison of variance explained
ax6 = plt.subplot(2, 3, 6)
methods = ['PCA', 'SVD', 'LDA']
variances = [
    sum(pca.explained_variance_ratio_),
    sum(svd.explained_variance_ratio_),
    sum(lda.explained_variance_ratio_)
]
bars = ax6.bar(methods, variances, color=['blue', 'green', 'orange'], alpha=0.7)
ax6.set_ylabel('Total Variance Explained')
ax6.set_title('Comparison of Methods (2 components)')
ax6.set_ylim([0, 1])
ax6.grid(True, alpha=0.3, axis='y')

for bar, var in zip(bars, variances):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{var:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ==================== CLASSIFICATION PERFORMANCE ====================
print("\n" + "="*70)
print("CLASSIFICATION PERFORMANCE COMPARISON")
print("="*70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_iris_scaled, y_iris, 
                                                     test_size=0.3, random_state=42)

# Original features
clf_original = LogisticRegression(max_iter=1000)
clf_original.fit(X_train, y_train)
acc_original = accuracy_score(y_test, clf_original.predict(X_test))

# PCA features
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
clf_pca = LogisticRegression(max_iter=1000)
clf_pca.fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))

# LDA features
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)
clf_lda = LogisticRegression(max_iter=1000)
clf_lda.fit(X_train_lda, y_train)
acc_lda = accuracy_score(y_test, clf_lda.predict(X_test_lda))

print(f"\nAccuracy with original features (4D): {acc_original:.4f}")
print(f"Accuracy with PCA features (2D): {acc_pca:.4f}")
print(f"Accuracy with LDA features (2D): {acc_lda:.4f}")

print("\n" + "="*70)
print("KEY DIFFERENCES:")
print("="*70)
print("PCA: Unsupervised, maximizes variance, ignores class labels")
print("SVD: Matrix factorization, similar to PCA but different computation")
print("LDA: Supervised, maximizes class separability, uses class labels")
print("="*70)