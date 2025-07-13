import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("students.csv")

# 2. Split into X and y
X = df[['Hours_Studied']]  # input feature
y = df['Score']            # output label

# 3. Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create the Linear Regression model
model = LinearRegression()

# 5. Train (fit) the model
model.fit(X_train, y_train)

# 6. Predict the scores for test data
predictions = model.predict(X_test)

# 7. Print predicted vs actual
print("Predicted Scores:", predictions)
print("Actual Scores:   ", y_test.values)

# 8. Visualize: draw the best fit line
plt.scatter(X, y, color='blue', label='Actual Scores')
plt.plot(X, model.predict(X), color='red', label='Prediction Line')
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Hours vs Score - Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
