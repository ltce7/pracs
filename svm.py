import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate sample classification data
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Dataset shape:", X.shape)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("\n" + "="*60)

# ==================== LINEAR SVM ====================
print("\n1. LINEAR SVM")
print("="*60)

svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)

print(f"Linear SVM Accuracy: {accuracy_linear:.4f}")
print(f"Number of support vectors: {len(svm_linear.support_vectors_)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_linear))

# ==================== RBF (Gaussian) KERNEL SVM ====================
print("\n2. RBF KERNEL SVM")
print("="*60)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"RBF SVM Accuracy: {accuracy_rbf:.4f}")
print(f"Number of support vectors: {len(svm_rbf.support_vectors_)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rbf))

# ==================== POLYNOMIAL KERNEL SVM ====================
print("\n3. POLYNOMIAL KERNEL SVM")
print("="*60)

svm_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)

print(f"Polynomial SVM Accuracy: {accuracy_poly:.4f}")
print(f"Number of support vectors: {len(svm_poly.support_vectors_)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_poly))

# ==================== VISUALIZATION ====================
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', alpha=0.7)
    
    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, linewidth=1.5, facecolors='none', edgecolors='green', 
               label='Support Vectors')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()

# Create subplots for different kernels
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Linear SVM
plt.subplot(2, 2, 1)
plot_decision_boundary(svm_linear, X_train, y_train, 
                      f'Linear SVM (Accuracy: {accuracy_linear:.4f})')

# RBF SVM
plt.subplot(2, 2, 2)
plot_decision_boundary(svm_rbf, X_train, y_train, 
                      f'RBF SVM (Accuracy: {accuracy_rbf:.4f})')

# Polynomial SVM
plt.subplot(2, 2, 3)
plot_decision_boundary(svm_poly, X_train, y_train, 
                      f'Polynomial SVM (Accuracy: {accuracy_poly:.4f})')

# Accuracy comparison
plt.subplot(2, 2, 4)
kernels = ['Linear', 'RBF', 'Polynomial']
accuracies = [accuracy_linear, accuracy_rbf, accuracy_poly]
bars = plt.bar(kernels, accuracies, color=['blue', 'green', 'orange'], alpha=0.7)
plt.ylabel('Accuracy')
plt.title('SVM Kernels Comparison')
plt.ylim([0.8, 1.0])
plt.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Linear SVM:     {accuracy_linear:.4f}")
print(f"RBF SVM:        {accuracy_rbf:.4f}")
print(f"Polynomial SVM: {accuracy_poly:.4f}")