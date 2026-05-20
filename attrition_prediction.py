import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("HR_Employee_Attrition.csv")

# Display dataset info
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

# Encode all text columns
label_encoder = LabelEncoder()



# Convert categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
# Features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy * 100)

# Add prediction column
df["Prediction"] = model.predict(X)

# Save output file
df.to_csv("Predicted_Attrition.csv", index=False)

print("\nPredicted_Attrition.csv created successfully!")

# Feature Importance
importance = model.feature_importances_

# Plot graph
plt.figure(figsize=(10,6))
plt.barh(X.columns, importance)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")

plt.show()