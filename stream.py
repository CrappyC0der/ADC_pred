# Let's create a sample Streamlit app script for this use case

streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('new_data.csv')

# Drop unnecessary columns
data = data.drop(columns=['id', 'full_name'])

# Fill missing values
data.fillna('Unknown', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target
X = data.drop(columns=['click'])
y = data['click']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

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
"""

# Save the Streamlit app script to a .py file
streamlit_file_path = '/mnt/data/ad_click_prediction_app.py'
with open(streamlit_file_path, 'w') as file:
    file.write(streamlit_code)

streamlit_file_path  # Return the file path for download





