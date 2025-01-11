import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate random data points
np.random.seed(42)
X_train = np.linspace(0, 10, 100)
y_train = 2 * X_train + np.random.randn(*X_train.shape) * 2  # True slope = 2, b = 0, with random noise

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Data Points')
plt.legend()
plt.show()

# Step 2: Define the cost function
def cost_function(w, X, y):
    m = len(y)
    predictions = w * X
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Compute cost for different values of w
w_values = np.linspace(-10, 10, 100)
cost_values = [cost_function(w, X_train, y_train) for w in w_values]

# Plot the cost function
plt.plot(w_values, cost_values, color='red', label='Cost Function')
plt.xlabel('w')
plt.ylabel('Cost')
plt.title('Cost Function vs Weight')
plt.legend()
plt.show()

# Step 3: Implement Gradient Descent
def gradient_descent(X, y, alpha, iterations):
    m = len(y)
    w = 0  # Initial weight
    cost_history = []
    
    for i in range(iterations):
        # Calculate predictions and error
        predictions = w * X
        error = predictions - y
        gradient = (1 / m) * np.dot(X, error)  # Gradient of cost w.r.t. w
        w -= alpha * gradient  # Update rule for w
        
        # Compute cost and store in history
        cost = cost_function(w, X, y)
        cost_history.append(cost)
        
        # Visualize the line at every 10% of iterations
        if i % (iterations // 10) == 0:
            plt.scatter(X, y, color='blue', label='Training Data')
            plt.plot(X, w * X, color='red', label=f'Iteration {i}: w = {w:.4f}')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.title(f'Gradient Descent Progress\nIteration {i}, Cost = {cost:.4f}')
            plt.legend()
            plt.show()

    return w, cost_history

# Ask user for learning rate and number of iterations
alpha = float(input("Enter the learning rate (alpha): "))
iterations = int(input("Enter the number of iterations: "))

# Run gradient descent
optimal_w, cost_history = gradient_descent(X_train, y_train, alpha, iterations)

# Step 4: Plot the final regression line
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, optimal_w * X_train, color='green', label=f'Optimal Line: y = {optimal_w:.4f}x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Final Regression Line')
plt.legend()
plt.show()

print(f"Optimal weight (w) that minimizes the cost function: {optimal_w}")