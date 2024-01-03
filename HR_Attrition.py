import streamlit as st
import pandas as pd
import time
import joblib

with st.spinner('Fetching Latest ML Model'):
    # Create the model, including its weights and the optimizer
    model = joblib.load("HRAttritionModel.pkl")
    time.sleep(1)
    st.success('Model Loaded!')


st.title('HR Attrition App \n\n')
st.subheader('Input values.') 