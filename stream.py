import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model
model_file = 'Trained_model.sav'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Load data for reference to match the feature names (assuming 'new_data.csv' is present)
data = pd.read_csv('new_data.csv')

# Drop unnecessary columns
data = data.drop(columns=['id', 'full_name'])

# Fill missing values
data.fillna('Unknown', inplace=True)

# One-Hot Encoding categorical variables (this assumes that the same encoding was used during training)
encoded_data = pd.get_dummies(data, columns=['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day'])

# Get the final list of feature names that were used in the model
feature_columns = encoded_data.columns.drop('click')

# Streamlit App Interface
st.title('Ad Click Prediction')

# Input fields
age = st.slider('Age', min_value=18, max_value=80, value=25)
gender = st.selectbox('Gender', ['Female', 'Male', 'Non-Binary', 'Unknown'])
device_type = st.selectbox('Device Type', ['Desktop', 'Mobile', 'Tablet', 'Unknown'])
ad_position = st.selectbox('Ad Position', ['Bottom', 'Side', 'Top', 'Unknown'])
browsing_history = st.selectbox('Browsing History', ['Education', 'Entertainment', 'News', 'Shopping', 'Social Media', 'Unknown'])
time_of_day = st.selectbox('Time of Day', ['Afternoon', 'Evening', 'Morning', 'Night', 'Unknown'])

# Create a DataFrame for the user input and apply One-Hot Encoding to match the training format
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'device_type': [device_type],
    'ad_position': [ad_position],
    'browsing_history': [browsing_history],
    'time_of_day': [time_of_day]
})

# Apply One-Hot Encoding to the input data
input_data_encoded = pd.get_dummies(input_data)

# Ensure that the input has the same columns as the training data by reindexing the columns
input_data_encoded = input_data_encoded.reindex(columns=feature_columns, fill_value=0)

# Make prediction
prediction = model.predict(input_data_encoded)

# Display result
if prediction[0] == 1:
    st.success("The user is predicted to click on the ad!")
else:
    st.error("The user is predicted to not click on the ad.")
