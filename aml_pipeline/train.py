# train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate and save the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")

os.makedirs('outputs', exist_ok=True)
joblib.dump(model, 'outputs/iris_model.joblib')
