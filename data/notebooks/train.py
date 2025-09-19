
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 1. Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2
)

# 3. Pipeline (scaler + KNN)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])

# 4. Hyperparameter tuning
param_grid = {
    'clf__n_neighbors': [3, 5, 7, 9],
    'clf__weights': ['uniform', 'distance']
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

import os

# 5. Save best model safely
# Get absolute path to project root (2 levels up from notebooks/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Make sure models folder exists
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Save the model inside models/
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_model.pkl")
joblib.dump(grid.best_estimator_, MODEL_PATH)

print("✅ Model saved to:", MODEL_PATH)

