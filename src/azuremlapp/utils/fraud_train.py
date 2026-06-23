# =============================
# Import Libraries
# =============================
import os
import pandas as pd
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# =============================
# Load .env file
# =============================
load_dotenv()

file_path = os.getenv("file_path")
model_path = os.getenv("model_path")

# =============================
# Load Dataset
# =============================
df = pd.read_csv(file_path)
df.dropna(inplace=True)

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
# Train Model
# =============================
model = DecisionTreeClassifier(
    max_depth=4,
    class_weight="balanced",
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
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# =============================
# Create Artifact (Model Package)
# =============================
artifact = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
}

# =============================
# Save Model Artifact
# =============================
joblib.dump(artifact, model_path)

print(f"\n✅ Model saved successfully at: {model_path}")
