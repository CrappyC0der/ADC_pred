# %%
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# %%
loaded_model = pickle.load(open("Trained_model.sav",'rb'))

# %%
def Adc_pred(input_data):
    inp_d_arr = np.asarray(input_data)
    inp_d_arr_r = inp_d_arr.reshape(1,-1)
    prediction = loaded_model.predict(inp_d_arr_r)

    if prediction[0] == 0 :
        return "User will click on the ad"
    elif prediction[0] == 1:
        return "User won't click on the ad"


# %%
def main() :
    st.title("AdClicker Prediction")

    age = st.text_input("Enter Age : ")
    gender = st.selectbox("Gender : ",["Male","Female","Non-Binary","Unknown"])
    device_type = st.selectbox("Enter Device : ",["Desktop","Mobile","Mobile","Tablet","Unknown"])
    browsing_history = st.selectbox("Browsing history : ",["Shopping","Education","Entertainment","Social media","News","Unknown"])
    time_of_day = st.selectbox("Time of the daye : ",["Morning","Afternoon","Evening","Unknown"])
    click = ''

    if st.button("Predict"):
        click = Adc_pred([age,gender,device_type,browsing_history,time_of_day])

    st.success(click)


# %%




