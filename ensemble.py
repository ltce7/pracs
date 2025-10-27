import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Generate sample classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset shape:", X.shape)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("\n" + "="*60)

# ==================== BAGGING ====================
print("\n1. BAGGING CLASSIFIER")
print("="*60)

# Bagging with Decision Trees
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    random_state=42
)
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

print(f"Bagging Accuracy: {accuracy_bagging:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_bagging))

# ==================== RANDOM FOREST (Advanced Bagging) ====================
print("\n2. RANDOM FOREST (Advanced Bagging)")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print("\nTop 5 Feature Importances:")
feature_importance = rf_model.feature_importances_
top_indices = np.argsort(feature_importance)[-5:][::-1]
for idx in top_indices:
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")

# ==================== BOOSTING - ADABOOST ====================
print("\n3. ADABOOST (Boosting)")
print("="*60)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
adaboost_model.fit(X_train, y_train)
y_pred_ada = adaboost_model.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)

print(f"AdaBoost Accuracy: {accuracy_ada:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ada))

# ==================== BOOSTING - GRADIENT BOOSTING ====================
print("\n4. GRADIENT BOOSTING (Boosting)")
print("="*60)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

print(f"Gradient Boosting Accuracy: {accuracy_gb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

# ==================== COMPARISON VISUALIZATION ====================
models = ['Bagging', 'Random Forest', 'AdaBoost', 'Gradient Boosting']
accuracies = [accuracy_bagging, accuracy_rf, accuracy_ada, accuracy_gb]

plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Ensemble Learning Methods Comparison')
plt.ylim([0.8, 1.0])
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Bagging:           {accuracy_bagging:.4f}")
print(f"Random Forest:     {accuracy_rf:.4f}")
print(f"AdaBoost:          {accuracy_ada:.4f}")
print(f"Gradient Boosting: {accuracy_gb:.4f}")