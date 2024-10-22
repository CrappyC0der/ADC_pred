import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.set_page_config(page_title="Ad Click Dataset Analysis")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("ad_click_dataset.csv")
    return df

df = load_data()

# Title
st.title("Ad Click Dataset Analysis")

# Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(df)

# Data info
st.subheader("Dataset Information")
st.write(df.info())

# Null percentages
st.subheader("Null Percentages")

def get_null_percentages(df):
    result = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        total_count = len(df)
        null_percentage = (null_count / total_count) * 100
        result[col] = round(null_percentage, 2)
    return result

null_percentages = get_null_percentages(df)
st.write(null_percentages)

# Age distribution
st.subheader("Age Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['age'], bins=10, kde=True, ax=ax)
ax.set_title('Age Distribution')
st.pyplot(fig)

# Gender distribution
st.subheader("Gender Distribution")
gender_counts = df['gender'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
gender_counts.plot(kind='bar', ax=ax)
ax.set_title('Gender Distribution')
st.pyplot(fig)

# Device type distribution
st.subheader("Device Type Distribution")
device_counts = df['device_type'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
device_counts.plot(kind='bar', ax=ax)
ax.set_title('Device Type Distribution')
st.pyplot(fig)

# Click rate by gender
st.subheader("Click Rate by Gender")
click_rate_gender = df.groupby('gender')['click'].mean()
fig, ax = plt.subplots(figsize=(10, 6))
click_rate_gender.plot(kind='bar', ax=ax)
ax.set_title('Click Rate by Gender')
ax.set_ylabel('Click Rate')
st.pyplot(fig)

# Click rate by device type
st.subheader("Click Rate by Device Type")
click_rate_device = df.groupby('device_type')['click'].mean()
fig, ax = plt.subplots(figsize=(10, 6))
click_rate_device.plot(kind='bar', ax=ax)
ax.set_title('Click Rate by Device Type')
ax.set_ylabel('Click Rate')
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# Add more visualizations and analysis as needed

st.write("This is a basic Streamlit app for the Ad Click Dataset analysis. You can expand on this by adding more visualizations, insights, and interactive elements.")




