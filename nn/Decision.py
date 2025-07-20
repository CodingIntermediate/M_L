from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
# Step 1: Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create the model
model = DecisionTreeClassifier()

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy:.2f}")
# Accuracy:1.00

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add target column to DataFrame
iris_df['target'] = iris.target
# View the first few rows
print(iris_df.head())

plt.figure(figsize=(10,8))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
print("features:",iris.feature_names)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show() 
# (graph)
