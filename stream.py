
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model_file = 'Trained_model.sav'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title("Ad Click Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Assuming the dataset requires standardization or specific preprocessing
    st.write("Processing the data...")
    # Apply any necessary data preprocessing here (scaling as an example)
    scaler = StandardScaler()
    features = data.iloc[:, :-1]  # Exclude label/target column if present
    scaled_features = scaler.fit_transform(features)
    
    # Prediction
    if st.button("Predict Ad Clicks"):
        predictions = model.predict(scaled_features)
        st.write("Predictions:", predictions)

    st.write("Prediction Completed.")

else:
    st.write("Please upload a CSV file to start prediction.")




