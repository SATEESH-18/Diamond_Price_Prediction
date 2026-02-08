import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from feature_engine.outliers import Winsorizer

# Load data
df = pd.read_csv("diamonds.csv")

# Drop unnecessary columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.drop_duplicates()

# Split features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split (75:25 as per assignment)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Columns
numerical_cols = ["carat", "depth", "table", "x", "y", "z"]
categorical_cols = ["cut", "color", "clarity"]

# Preprocessing
numeric_pipeline = Pipeline(steps=[
    ("winsor", Winsorizer(capping_method="iqr", tail="both", fold=1.5)),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

# Model
knn = KNeighborsRegressor(
    n_neighbors=5,
    weights="distance",
    p=1
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", knn)
])

# Train
pipeline.fit(X_train, y_train)

# Evaluation
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

print("Training R2 Score:", r2_score(y_train, y_train_pred))
print("Testing R2 Score:", r2_score(y_test, y_test_pred))

# Save model
with open("diamond_knn_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved as diamond_knn_model.pkl")