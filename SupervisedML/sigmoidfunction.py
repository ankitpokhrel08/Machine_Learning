import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate binary classification data
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=0)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
X_test = np.linspace(-3, 3, 300)

# Plot data and model with transition
plt.scatter(X, y, color='blue', label='Data')
for i in range(1, 11):
    alpha = i / 10.0
    y_prob = sigmoid(X_test * model.coef_ * alpha + model.intercept_ * alpha).ravel()
    plt.plot(X_test, y_prob, color='red', alpha=alpha, label=f'alpha={alpha:.1f}' if i == 10 else "")

# Calculate and plot decision boundary
decision_boundary = -model.intercept_ / model.coef_
plt.axvline(x=decision_boundary, color='green', linestyle='--', label='Decision Boundary')

plt.xlabel('Feature')
plt.ylabel('Probability')
plt.title('Sigmoid Function, Logistic Regression Model Transition, and Decision Boundary')
plt.legend()
plt.show()
