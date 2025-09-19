import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load(r"c:\iris-classifier\models\iris_model.pkl")
  # ✅ only 1 level up
print("✅ Model loaded from: ../models/iris_model.pkl")
0.6
# 2. Iris target names
iris_classes = ["Setosa", "Versicolor", "Virginica"]

# 3. Take user input
print("\nEnter flower measurements:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width  = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width  = float(input("Petal width (cm): "))

# 4. Prepare input
sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# 5. Make prediction
prediction = model.predict(sample)[0]
print(f"\n🌼 Predicted Iris species: {iris_classes[prediction]}")

