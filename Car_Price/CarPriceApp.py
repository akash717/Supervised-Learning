# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

scaled = pd.read_csv("C:/Users/akash/scaled_car_price.csv")
df = pd.read_csv("C:/Users/akash/car_price.csv")

encodings = {
    "fueltype": {"Diesel": 1, "Gas": 0},
    "4wd": {"Yes": 1, "No": 0},
    "fwd": {"Yes": 1, "No": 0},
}


def modelPredict(MPG, fueltype, wheelbase,
                 CarCompanyName, fourWheelDrive, FrontWheelDrive, cylinder):
    with open("car_price.pkl", 'rb') as file:
        reg_model = pickle.load(file)
    scaled = pd.read_csv("scaled_car_price.csv")
    preScaled = pd.read_csv("prescale_car_price.csv")
    df = pd.read_csv("car_price.csv")
    new_features = [list(preScaled.mean())]
    new_features[0][1] = wheelbase
    new_features[0][3] = cylinder
    new_features[0][6] = df[df['CarCompanyName']
                            == CarCompanyName]['price'].mean()
    new_features[0][7] = fueltype
    new_features[0][10] = fourWheelDrive
    new_features[0][11] = FrontWheelDrive
    new_features[0][12] = MPG
    print(new_features)
    scaler = StandardScaler()
    new_features = scaler.fit_transform(
        np.vstack((preScaled, new_features)))[-1]
    print(new_features)
    price = reg_model.predict(new_features.reshape(1, 13))[0]
    return "$" + str(round(price, 2))


# Layout
st.set_page_config(layout='wide')


fueltype = st.sidebar.radio(
    "Transmission Type",
    ("Diesel", "Gas")
)

fourWheelDrive = st.sidebar.radio(
    "Four Wheel Drive",
    ("Yes", "No")
)

FrontWheelDrive = st.sidebar.radio(
    "Front Wheel Drive",
    ("Yes", "No")
)

CarCompanyName = st.sidebar.selectbox(
    "Car Company Name",
    df['CarCompanyName'].unique()
)


cylinder = int(st.sidebar.selectbox(
    "Number of Cylinder",
    ("1", "2", "3", "4", "5", "6"))
)


st.write('''
         ## Price Prediction in USD
         ### Glimpse of how historic data looks
         ''')

st.dataframe(df.head())

MPG = st.slider("Set Mileage of the Car",
                5, 50, 5)

wheelbase = st.slider("WheelBase of the Car", 130, 200, 130)

if st.button('Predict Price'):
    fueltype = encodings['fueltype'][fueltype]
    fourWheelDrive = encodings['4wd'][fourWheelDrive]
    FrontWheelDrive = encodings['fwd'][FrontWheelDrive]
    price = modelPredict(MPG, fueltype, wheelbase,
                         CarCompanyName, fourWheelDrive, FrontWheelDrive, cylinder)
    st.text("Predicted Price for car is "+str(price))
