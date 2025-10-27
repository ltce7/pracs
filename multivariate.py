import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate sample data with multiple features
np.random.seed(42)
n_samples = 200
X1 = np.random.rand(n_samples) * 10
X2 = np.random.rand(n_samples) * 5
X3 = np.random.rand(n_samples) * 8

# Create target variable: y = 3*X1 + 2*X2 - 1.5*X3 + 10 + noise
y = 3*X1 + 2*X2 - 1.5*X3 + 10 + np.random.randn(n_samples) * 2

# Combine features into a matrix
X = np.column_stack([X1, X2, X3])

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
df['Target'] = y

print("Dataset Info:")
print(df.head())
print(f"\nDataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMultivariate Linear Regression Results:")
print(f"Coefficients: {model.coef_}")
print(f"Feature1 coefficient: {model.coef_[0]:.4f}")
print(f"Feature2 coefficient: {model.coef_[1]:.4f}")
print(f"Feature3 coefficient: {model.coef_[2]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Multivariate Linear Regression: Predicted vs Actual')
plt.grid(True)
plt.show()

# Feature importance visualization
plt.figure(figsize=(10, 6))
features = ['Feature1', 'Feature2', 'Feature3']
plt.bar(features, model.coef_)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Feature Coefficients')
plt.grid(True, axis='y')
plt.show()