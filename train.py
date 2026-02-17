import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv('taxi_trip_pricing.csv')


# Drop rows where target is missing
df = df.dropna(subset=['Trip_Price'])

print("Remaining rows:", df.shape[0])

# Log transform
df['Trip_Distance_km'] = np.log1p(df['Trip_Distance_km'])
df['Trip_Price'] = np.log1p(df['Trip_Price'])

X = df.drop('Trip_Price', axis=1)
y = df['Trip_Price']

categorical_cols = ['Time_of_Day','Day_of_Week','Traffic_Conditions','Weather']
numeric_cols = X.select_dtypes(exclude='object').columns

# Preprocessing
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Final Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train
pipeline.fit(X, y)

# Save Model
with open('taxi_price_model.pkl','wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model saved successfully!")
