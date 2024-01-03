import streamlit as st
import pandas as pd
import time
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

with st.spinner('Fetching Latest ML Model'):
    # Create the model, including its weights and the optimizer
    model = joblib.load("HRAttritionModel.pkl")
    time.sleep(1)
    st.success('Model Loaded!')


st.title('HR Attrition App \n\n')
st.subheader('Input values.') 
