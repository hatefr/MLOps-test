import os

from joblib import dump
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
iris = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Define a list of models to experiment with
model = DecisionTreeClassifier(max_depth=2)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# Create a model folder w=if it does not exist
if not os.path.exists("models"):
    os.makedirs("models")

# dump(model, "models/model.joblib")
model.save("models/model.pkl")
