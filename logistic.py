import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

# Generate sample classification data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"\nCoefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"\nConfusion Matrix:")
print(conf_matrix)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize decision boundary
plt.figure(figsize=(12, 5))

# Plot 1: Data points with decision boundary
plt.subplot(1, 2, 1)
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
           edgecolors='black', alpha=0.7, label='Training data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', 
           marker='s', edgecolors='black', alpha=0.9, label='Testing data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()

# Plot 2: Confusion Matrix
plt.subplot(1, 2, 2)
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'])
plt.yticks(tick_marks, ['Class 0', 'Class 1'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.tight_layout()
plt.show()