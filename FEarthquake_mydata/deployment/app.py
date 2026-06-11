import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


Folder_name="FEarthquake_mydata"
HF_username="hasilm1"
App_name="Earthquake_prediction"

Model_name="best_predict_earthquake_model_v1.joblib"
data_filename="earthquakes.csv"

# Download and load the model
model_path = hf_hub_download(repo_id=str(HF_username)+"/"+str(App_name), filename=Model_name) # enter the Hugging Face username here
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title(str(App_name)+" App")
st.write("""
This application predicts the data set provided.
Please enter the data below to get a prediction.
""")

# User inputs
day = st.number_input("Day", min_value=1, max_value=31, value=1)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
year = st.number_input("Year", min_value=1900, max_value=2099, value=2026)

time="1981-06-05 13:02:04.800000+00:00"
year=1981
month=6
day_of_year=156
hour=13
longitude=140.6
depth=68.2
mag=5.2

#df = df.dropna()
#X = df.drop(columns=[target_col,'id','time'])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'day_of_year': day_of_year,
    'month': month,
    'year': year,
    'hour':hour,
    'longitude':longitude,
    'depth':depth,
    'mag':mag
}])
input_data1 = pd.DataFrame([{
    'day_of_year': day_of_year,
    'month': month,
    'year': year
}])
# Prediction button
if st.button("Predict "):
    prediction = model.predict(input_data1)[0]
    st.success(f"The latitude: **{prediction}**")
