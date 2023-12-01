import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import lightgbm as lgb
import pickle
import shap
from streamlit_shap import st_shap

st.set_page_config(
page_title="Explaniability",
page_icon="💡",
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
X_test = data["X_test"]
y_pred_test = data['y_pred_test']
y_test = data['y_test']


#st.dataframe(X_test)




shap.initjs()
explainer = shap.Explainer(model)
shap_values = explainer(X_test)



def plot_scatter_with_error(model_name, model_error, y_actual, y_predicted):
    # Create a scatter plot using Plotly Express
    fig = px.scatter(x=y_actual, y=y_predicted)
    fig.update_traces(marker=dict(color='blue'), selector=dict(mode='markers'))

    # Update the layout with title and subtitle
    fig.update_layout(
        title=f'{model_name} Scatter Plot (Actual vs. Predicted)',
        title_x=0,  # Align title to the center
        title_y=0.95,  # Adjust the title position
        title_font_size=24,
        title_font_family='Arial',
        # Add subtitle
        annotations=[
            dict(
                x=1, y=-0.16,
                xref='paper', yref='paper',
                text=f'Average Error: {"${:,.02f}".format(model_error)}',
                showarrow=False,
                font=dict(family='Arial', size=16)
            )
        ],
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
    )

    return fig




st.write('# Model Performance')
fig = plot_scatter_with_error('LightGBM Regressor', np.sqrt(mean_squared_error(y_test, y_pred_test)), y_test, y_pred_test)
st.plotly_chart(fig)




st.write('# Global Explainability')
st.write("""Proficiently articulating the behavior and decisions of complex machine learning models is pivotal in ensuring transparency, trustworthiness, and practical deployment in real-world scenarios. 
        So, below you can see which patterns our LightGBM Regressor model has learned to predict car prices.""")


st.write('## Summary Plot')
st_shap(shap.summary_plot(shap_values, X_test), height=500)
st.write("""Above we can see different insights:

- Newest cars tend to have highest price
- Cars with higher engine capacity tend to have highest price
- Cars with lower mileage tend to have higher price
- If the car is not registerd we can see a negative impact in their price
- If the car is diesel this feature tend also to slightly increase the price""")

st.write('## Average contribution of each feature to the car price')

st_shap(shap.plots.bar(shap_values), height=500)

st.write("""Here we can see that the two features with more impact on the car price are the year and the engine capacity of the car.""")





st.write('## Variables Deep Dive')

st.write('#### Mileage')

st_shap(shap.plots.scatter(shap_values[:,"mileage"]), height=500)

st.write("""Now related with the impact on the mileage feature on the price our model has learned that mileage has impact on the price in the following way. Mileage has impact increasing the price specially when mileage is 0 or very close to 0.""")

st.write('#### Engine Capacity')

st_shap(shap.plots.scatter(shap_values[:,"engV"]), height=500)

st.write("""Here we can see that the shap value is bigger as bigger is the engine capacity. So, as bigger the engine capacity bigger the price following a linear pattern.""")

st.write('#### Car year')

st_shap(shap.plots.scatter(shap_values[:,"year"]), height=500)
st.write("""For car year we can see that the shap value grows exponentially as the year grows. Meaning that after 2010 the impact on the price increase really fast.""")



st.write('## Relationship between variables')

st.write('#### Evolution of shap values of engine capacity based on mileage')
st_shap(shap.dependence_plot("engV", shap_values.values, X_test, interaction_index= "mileage"), height=500)

st.write('#### Evolution of shap values of car year based on mileage')
st_shap(shap.dependence_plot("year", shap_values.values, X_test, interaction_index= "mileage"), height=500)



st.write('hola')
st_shap(shap.plots.force(shap_values))
