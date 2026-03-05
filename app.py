import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.title("🚗 BMW Global Sales Prediction App")

st.write("Predict BMW Units Sold using ML Models (KNN, ANN, CNN)")

# Load models
knn = pickle.load(open("knn_MODEL.pkl","rb"))
ann = load_model("ann_model.h5")
cnn = load_model("cnn_model.h5")

# Input fields
year = st.number_input("Enter Year", 2018, 2030)
region = st.selectbox("Region", ["Asia","Europe","North America"])
model = st.selectbox("Model", ["BMW X1","BMW X3","BMW X5","BMW 3 Series","BMW 5 Series"])
price = st.number_input("Price")
marketing = st.number_input("Marketing Spend")

# Encoding (same as training)
region_dict = {"Asia":0,"Europe":1,"North America":2}
model_dict = {
    "BMW X1":0,
    "BMW X3":1,
    "BMW X5":2,
    "BMW 3 Series":3,
    "BMW 5 Series":4
}

region = region_dict[region]
model = model_dict[model]

features = np.array([[year, region, model, price, marketing]])

# Prediction
if st.button("Predict Sales"):

    knn_pred = knn.predict(features)

    ann_pred = ann.predict(features)

    cnn_features = features.reshape(features.shape[0], features.shape[1], 1)
    cnn_pred = cnn.predict(cnn_features)

    st.subheader("Prediction Results")

    st.write("KNN Prediction:", int(knn_pred[0]))
    st.write("ANN Prediction:", int(ann_pred[0][0]))
    st.write("CNN Prediction:", int(cnn_pred[0][0]))
