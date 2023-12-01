import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import shap
from streamlit_shap import st_shap

st.set_page_config(
page_title="Price Prediction",
page_icon="💰",
layout="wide",
initial_sidebar_state="expanded")

#Load the pickle file with the model and the label encoders
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

@st.cache_data
def load_data():
    df =  pd.read_csv("car_ad_display.csv", encoding = "ISO-8859-1", sep=";").drop(columns='Unnamed: 0')

    car_map = shorten_categories(df.car.value_counts(), 10)
    df['car'] = df['car'].map(car_map)

    model_map = shorten_categories(df.model.value_counts(), 10)
    df['model'] = df['model'].map(model_map)

    df = df[df["price"] <= 100000]
    df = df[df["price"] >= 1000]
    df = df[df["mileage"] <= 600]
    df = df[df["engV"] <= 7.5]
    df = df[df["year"] >= 1975]

    return df

df_original = load_data()

st.title("🔮 Car price Predictor 🔮")
st.write("""### Enter your car information to predict its price!""")

car = st.text_input('Car brand', value="Porshe")

body_types = (
    "crossover",
    "sedan",
    "van",
    "vagon",
    "hatch",
    "other",
)

body = st.selectbox("Body", body_types)

mileage = st.slider("Mileage", 0, 600, 80)

engV = st.slider("EngV", 0.0, 7.0, 3.5)

engType_types = (
    "Gas",
    "Petrol",
    "Diesel",
    "Other",
)
engType = st.selectbox("EngType", engType_types)

registered = st.radio(
    "Is it registered?",
    ('Yes', 'No'))

year  = st.slider("Year", 1975, 2015, 2010)

drive_types = (
    "full",
    "rear",
    "front",
)
drive = st.selectbox("Drive", drive_types)

yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']


ok = st.button("Calculate Price")
if ok:
    X_sample = np.array([[car, body, mileage, engV, engType, registered, year, drive ]])
    # Apply the encoder and data type corrections:
    X_sample[:, 0] = str(X_sample[:, 0][0] if X_sample[:, 0][0] in list(df_original['car'].unique()) else 'Other')
    X_sample[:, 0] = le_car.transform(X_sample[:,0])
    X_sample[:, 1] = le_body.transform(X_sample[:,1])
    X_sample[:, 4] = le_engType.transform(X_sample[:,4])
    X_sample[:, 5] = int(1 if X_sample[:, 5][0] in yes_l else 0)
    X_sample[:, 7] = le_drive.transform(X_sample[:,7])

    X_sample = np.array([[
        int(X_sample[0, 0]), 
        int(X_sample[0, 1]), 
        int(X_sample[0, 2]), 
        float(X_sample[0, 3]), 
        int(X_sample[0, 4]), 
        int(X_sample[0, 5]), 
        int(X_sample[0, 6]), 
        int(X_sample[0, 7])
    ]])
   
    salary = model.predict(X_sample)
    st.subheader(f"The estimated price is ${salary[0]:.2f}")
    
    
    
    X_sample_d = X_sample[0]
    
    X_sample_df = pd.DataFrame({'car': X_sample_d[0], 
                                'body': X_sample_d[1], 
                                'mileage': X_sample_d[2], 
                                'engV': X_sample_d[3], 
                                'engType': X_sample_d[4], 
                                'registered': X_sample_d[5], 
                                'year': X_sample_d[6], 
                                'drive': X_sample_d[7]}, index=[0])

    shap.initjs()
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample_df)
    
    
    st.write('# Local Explainability')
    st.write('## How the model estimated your car price')
    st.write('### Force plot')
    st_shap(shap.plots.force(shap_values[0]), height=150)
    st.write('### Decision plot')
    st_shap(shap.decision_plot(shap_values[0].base_values,shap_values[0].values, X_sample_df), height=500)

