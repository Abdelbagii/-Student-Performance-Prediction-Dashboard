import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Create performance label from final grade G3
def categorize_performance(grade):
    if grade >= 15:
        return "High"
    elif grade >= 10:
        return "Medium"
    else:
        return "Low"

df["performance"] = df["G3"].apply(categorize_performance)

# Drop original target columns
X = df.drop(columns=["G3", "performance"])
y = df["performance"]

# Encode categorical columns
label_encoders = {}
for column in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Save files
joblib.dump(model, "model/student_performance_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")

print("\nModel and encoders saved successfully.")