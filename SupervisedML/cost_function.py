import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

# ...existing code...

# Generate some sample training data
X_train = np.linspace(0, 10, 100)
y_train = 2 * X_train + 1 + np.random.randn(*X_train.shape) * 2

# Plot the training set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_train, y_train, color='blue', label='Training data')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Training Set')
ax1.legend()

# Define the cost function
def cost_function(w, X, y):
    m = len(y)
    predictions = w * X
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Generate values for w and compute the cost for each
w_values = np.linspace(-10, 10, 100)
cost_values = [cost_function(w, X_train, y_train) for w in w_values]

# Plot the cost function
line, = ax2.plot(w_values, cost_values, color='red', label='Cost function')
ax2.set_xlabel('w')
ax2.set_ylabel('Cost')
ax2.set_title('Cost Function')
ax2.legend()

# Add a cursor to the cost function plot
cursor = Cursor(ax2, useblit=True, color='green', linewidth=1)

# Function to update the training set plot with the selected weight
def on_click(event):
    if event.inaxes == ax2:
        w = event.xdata
        y_pred = w * X_train
        
        # Create a new figure for the selected weight
        fig_new, ax_new = plt.subplots(figsize=(6, 5))
        ax_new.scatter(X_train, y_train, color='blue', label='Training data')
        ax_new.plot(X_train, y_pred, color='red', label=f'Prediction (w={w:.2f})')
        ax_new.set_xlabel('X')
        ax_new.set_ylabel('y')
        ax_new.set_title(f'f(x) = {w:.2f} * x')
        ax_new.legend()
        plt.show()

# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()

# ...existing code...
