import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

st.title("🚗 BMW Global Sales Prediction")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("bmw_global_sales_2018_2025.csv")

# Encode categorical columns
df["Region"] = df["Region"].astype("category").cat.codes
df["Model"] = df["Model"].astype("category").cat.codes

X = df.drop("Units_Sold", axis=1)
y = df["Units_Sold"]

# -------------------------------
# Train model if file not found
# -------------------------------
model_path = "knn_model.pkl"

if os.path.exists(model_path):
    knn = pickle.load(open(model_path, "rb"))
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    pickle.dump(knn, open(model_path, "wb"))

# -------------------------------
# User Inputs
# -------------------------------
year = st.number_input("Year", 2018, 2030)
region = st.selectbox("Region", ["Asia", "Europe", "North America"])
model = st.selectbox("Model", ["BMW X1", "BMW X3", "BMW X5", "BMW 3 Series", "BMW 5 Series"])
price = st.number_input("Price")
marketing = st.number_input("Marketing Spend")

# Encoding input
region_map = {"Asia":0, "Europe":1, "North America":2}
model_map = {
    "BMW X1":0,
    "BMW X3":1,
    "BMW X5":2,
    "BMW 3 Series":3,
    "BMW 5 Series":4
}

region = region_map[region]
model = model_map[model]

features = np.array([[year, region, model, price, marketing]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sales"):

    prediction = knn.predict(features)

    st.success(f"Predicted BMW Units Sold: {int(prediction[0])}")
