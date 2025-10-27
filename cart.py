import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset Information:")
print(f"Features: {iris.feature_names}")
print(f"Target classes: {iris.target_names}")
print(f"Dataset shape: {X.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the CART (Decision Tree) model
cart_model = DecisionTreeClassifier(
    criterion='gini',  # Can also use 'entropy' for information gain
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
cart_model.fit(X_train, y_train)

# Make predictions
y_pred = cart_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nCART Algorithm Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Tree depth: {cart_model.get_depth()}")
print(f"Number of leaves: {cart_model.get_n_leaves()}")
print(f"\nFeature Importances:")
for feature, importance in zip(iris.feature_names, cart_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

print(f"\nConfusion Matrix:")
print(conf_matrix)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(cart_model, 
          filled=True, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          rounded=True,
          fontsize=10)
plt.title('CART Decision Tree Visualization')
plt.show()

# Feature importance visualization
plt.figure(figsize=(10, 6))
features = iris.feature_names
importances = cart_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances in CART')
plt.tight_layout()
plt.show()

# Confusion matrix visualization
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

for i in range(len(iris.target_names)):
    for j in range(len(iris.target_names)):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.tight_layout()
plt.show()