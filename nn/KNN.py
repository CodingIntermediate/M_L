import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create the KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plotting the Test Results
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_test[:, 2], X_test[:, 3], c=y_pred, cmap='viridis', edgecolor='k', s=100)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("KNN Predictions (Petal Length vs Width)")
plt.colorbar(scatter, ticks=[0, 1, 2], label='Predicted Class')
plt.grid(True)
plt.show()

# (graph)
