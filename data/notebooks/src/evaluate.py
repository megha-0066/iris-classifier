import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib, os
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)
print("✅ Model loaded from:", MODEL_PATH)

# Load iris data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # ✅ Keep feature names
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2
)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("📊 Test Accuracy:", acc)

