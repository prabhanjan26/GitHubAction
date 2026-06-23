# =============================
# Import Libraries
# =============================
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================
# Load .env file
# =============================
load_dotenv()

# Get file path from .env
file_path = os.getenv("file_path")

# =============================
# Load Dataset
# =============================
df = pd.read_csv(file_path)

# print("✅ Dataset Loaded Successfully\n")
# print(df.head())
df.dropna(inplace= True)
# =============================
# Features & Target
# =============================
X = df.drop("fraud", axis=1)
y = df["fraud"]

# =============================
# Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# Train Decision Tree Model
# =============================
model = DecisionTreeClassifier(
    max_depth=4,
    class_weight="balanced",   # helps with fraud imbalance
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully")

# =============================
# Predictions
# =============================
y_pred = model.predict(X_test)

# =============================
# Evaluation
# =============================
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Evaluation Results:")
print(f"\nAccuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# # =============================
# # Feature Importance
# # =============================
# importance = pd.DataFrame({
#     "Feature": X.columns,
#     "Importance": model.feature_importances_
# }).sort_values(by="Importance", ascending=False)

# print("\n📈 Feature Importance:")
# print(importance)

# # =============================
# # Save Predictions (Optional)
# # =============================
# output = X_test.copy()
# output["Actual"] = y_test
# output["Predicted"] = y_pred

# output.to_csv("predictions_output.csv", index=False)

# print("\n✅ Predictions saved to predictions_output.csv")