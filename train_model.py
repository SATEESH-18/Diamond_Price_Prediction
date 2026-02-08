import pandas as pd
import pickle

from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

# Load data
df = pd.read_csv("diamonds.csv")
df = df.drop(columns=["Unnamed: 0"]).drop_duplicates()

num_cols = df.select_dtypes(include="number").columns

# Winsorizer
winsor = Winsorizer(capping_method="iqr", tail="both", fold=1.5)
df[num_cols] = winsor.fit_transform(df[num_cols])

X = df.drop("price", axis=1)
y = df["price"]

numerical_cols = ["carat", "depth", "table", "x", "y", "z"]
categorical_cols = ["cut", "color", "clarity"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

model = KNeighborsRegressor(n_neighbors=5, weights="distance", p=1)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X, y)

# Save FULL pipeline
with open("diamond_price_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Pipeline saved successfully")