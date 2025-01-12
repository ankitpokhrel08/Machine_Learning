import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function for logistic regression
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m) * ((-y.T @ np.log(h + epsilon)) - ((1 - y).T @ np.log(1 - h + epsilon)))
    return cost

# Custom input data
age = np.array([16, 25, 47, 52, 46, 56, 48, 55, 60, 62])
married = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1])
X = np.column_stack((np.ones(age.shape[0]), age, married))
y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Fit logistic regression model
model = LogisticRegression()
model.fit(X[:, 1:], y)

# Predict probabilities
X_test = np.linspace(min(age), max(age), 300)
X_test_full = np.column_stack((np.ones(X_test.shape[0]), X_test, np.ones(X_test.shape[0])))
y_prob = sigmoid(X_test_full @ np.append(model.intercept_, model.coef_).flatten())

# Calculate cost function
theta = np.append(model.intercept_, model.coef_).flatten()
cost = cost_function(X, y, theta)
print(f"Cost: {cost}")

# Plot data and model
plt.scatter(age, y, color='blue', label='Data')
plt.plot(X_test, y_prob, color='red', label='Logistic Regression Model')
plt.xlabel('Age')
plt.ylabel('Probability')
plt.title(f'Logistic Regression Model with Custom Data\nCost: {cost}')
plt.legend()
plt.show()
