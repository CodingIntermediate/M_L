import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Input data
X = np.array([[1], [2], [3], [4], [5]])  # Hours studied
y = np.array([25, 45, 50, 65, 75])      # Marks scored

# Step 2: Create the model
model = LinearRegression()

# Step 3: Train the model
model.fit(X, y)

# Step 4: Predict scores
predicted_scores = model.predict(X)

# Step 5: Plot the result
plt.scatter(X, y, color='blue', label='Actual Marks')
plt.plot(X, predicted_scores, color='red', label='Predicted Line')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Linear Regression - Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
