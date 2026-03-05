import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

st.title("BMW Sales Prediction")

# --------------------------------
# Load Dataset (safe loading)
# --------------------------------

file_path = "bmw_global_sales_2018_2025.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    st.warning("Dataset not found. Creating sample dataset.")

    df = pd.DataFrame({
        "Year":[2018,2019,2020,2021,2022],
        "Region":[0,1,2,0,1],
        "Model":[0,1,2,3,4],
        "Price":[35000,42000,50000,48000,55000],
        "Marketing":[20000,25000,22000,30000,27000],
        "Units_Sold":[1200,1500,1100,1800,1600]
    })

X = df.drop("Units_Sold", axis=1)
y = df["Units_Sold"]

# --------------------------------
# Load / Train Model
# --------------------------------

model_file = "knn_model.pkl"

if os.path.exists(model_file):
    knn = pickle.load(open(model_file,"rb"))
else:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train,y_train)

    pickle.dump(knn,open(model_file,"wb"))

# --------------------------------
# User Input
# --------------------------------

year = st.number_input("Year",2018,2030)
region = st.number_input("Region Code (0-Asia,1-Europe,2-USA)")
model = st.number_input("Model Code (0-4)")
price = st.number_input("Price")
marketing = st.number_input("Marketing Spend")

features = np.array([[year,region,model,price,marketing]])

if st.button("Predict Sales"):

    prediction = knn.predict(features)

    st.success(f"Predicted Units Sold: {int(prediction[0])}")
