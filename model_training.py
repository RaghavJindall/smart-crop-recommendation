import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load dataset
df = pd.read_csv("crop_recommendation.csv")

print("✅ Dataset loaded successfully")
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Split features and target
X = df.drop("label", axis=1)   # features
y = df["label"]                # target

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully")

# 5. Save trained model in same folder
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as crop_model.pkl")
