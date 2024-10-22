import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load saved model
model_file = 'Trained_model.sav'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Load data for label encoding (assuming 'new_data.csv' is present for reference)
data = pd.read_csv('new_data.csv')

# Drop unnecessary columns
data = data.drop(columns=['id', 'full_name'])

# Fill missing values
data.fillna('Unknown', inplace=True)

# Encode categorical variables using the same method as in training
label_encoders = {}
for column in ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Streamlit App Interface
st.title('Ad Click Prediction')

# Input fields
age = st.slider('Age', min_value=18, max_value=80, value=25)
gender = st.selectbox('Gender', label_encoders['gender'].classes_)
device_type = st.selectbox('Device Type', label_encoders['device_type'].classes_)
ad_position = st.selectbox('Ad Position', label_encoders['ad_position'].classes_)
browsing_history = st.selectbox('Browsing History', label_encoders['browsing_history'].classes_)
time_of_day = st.selectbox('Time of Day', label_encoders['time_of_day'].classes_)

# Convert input to the same format as training data
input_data = pd.DataFrame({
    'age': [age],
    'gender': [label_encoders['gender'].transform([gender])[0]],
    'device_type': [label_encoders['device_type'].transform([device_type])[0]],
    'ad_position': [label_encoders['ad_position'].transform([ad_position])[0]],
    'browsing_history': [label_encoders['browsing_history'].transform([browsing_history])[0]],
    'time_of_day': [label_encoders['time_of_day'].transform([time_of_day])[0]]
})

# Make prediction
prediction = model.predict(input_data)

# Display result
if prediction[0] == 1:
    st.success("The user is predicted to click on the ad!")
else:
    st.error("The user is predicted to not click on the ad.")
