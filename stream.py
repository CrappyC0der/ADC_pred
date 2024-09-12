{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a1f47a87-d72b-49a8-b347-5be1e883e9fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9332e0a9-903b-49c2-aed0-ccad04cf6590",
   "metadata": {},
   "outputs": [],
   "source": [
    "loaded_model = pickle.load(open(\"Trained_model.sav\",'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1015db46-58d0-4c8a-9c44-c937d487ed11",
   "metadata": {},
   "outputs": [],
   "source": [
    "def Adc_pred(input_data):\n",
    "    inp_d_arr = np.asarray(input_data)\n",
    "    inp_d_arr_r = inp_d_arr.reshape(1,-1)\n",
    "    prediction = loaded_model.predict(inp_d_arr_r)\n",
    "\n",
    "    if prediction[0] == 0 :\n",
    "        return \"User will click on the ad\"\n",
    "    elif prediction[0] == 1:\n",
    "        return \"User won't click on the ad\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "bd1511b6-6bc0-4839-91fe-02a13991dc8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main() :\n",
    "    st.title(\"AdClicker Prediction\")\n",
    "\n",
    "    age = st.text_input(\"Enter Age : \")\n",
    "    gender = st.selectbox(\"Gender : \",[\"Male\",\"Female\",\"Non-Binary\",\"Unknown\"])\n",
    "    device_type = st.selectbox(\"Enter Device : \",[\"Desktop\",\"Mobile\",\"Mobile\",\"Tablet\",\"Unknown\"])\n",
    "    browsing_history = st.selectbox(\"Browsing history : \",[\"Shopping\",\"Education\",\"Entertainment\",\"Social media\",\"News\",\"Unknown\"])\n",
    "    time_of_day = st.selectbox(\"Time of the daye : \",[\"Morning\",\"Afternoon\",\"Evening\",\"Unknown\"])\n",
    "    click = ''\n",
    "\n",
    "    if st.button(\"Predict\"):\n",
    "        click = Adc_pred([age,gender,device_type,browsing_history,time_of_day])\n",
    "\n",
    "    st.success(click)\n",
    "\n",
    "if __name__ ==\"__main___\" :\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa95fccb-e022-4aa0-becc-76b53d89274a",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
