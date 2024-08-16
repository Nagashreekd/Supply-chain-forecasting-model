# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:10:46 2023

@author: nagashree k d
"""

import pandas as pd
import streamlit as st
import numpy as np
from statsmodels.regression.linear_model import OLSResults
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model1.pickle")
hide_menu_style = """
<style>
    MainMenu { visibility: hidden; }
    div[data-testid="stHorizontalBLock"] > div:nth-child(1) {
        border: 2px solid #doe0db;
        border-radius: 5px;
        text-align: center;
        background: dodgerblue;
        padding: 25px;
    }
    div[data-testid="stHorizontalBLock"] > div:nth-child(2) {
        border: 2px solid #doe0db;
        border-radius: 5px;
        text-align: center;
        background: dodgerblue;
        padding: 25px;
    }
</style>
"""

main_title = """
<div>
    <h1 style='color: black;
    text-align: center;
    font-size: 35px;
    margin-top: -95px;'>
    </h1>
</div>
"""

data = pd.read_csv(r"C:\Users\nagashree k d\forecast_test.csv")
def main():
    st.set_page_config(page_title='Dashboard', layout='wide', initial_sidebar_state='auto')
    st.title("Forecasting")
    st.sidebar.title("Forecasting")
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.markdown(main_title, unsafe_allow_html=True)
    
    month = st.sidebar.slider('Months', 1, 12, 1)
    period = month * 30
    
    if st.button("Predict"):
        forecast_test = pd.DataFrame(model.predict(start=data.index[0], end=data.index[-1]))
        results = pd.concat([data, forecast_test], axis=1)
        
        st.text(" ")
        st.subheader('Plot forecasts against actual outcomes', anchor=None)
        # plot forecasts against actual outcomes
        fig, ax = plt.subplots()
        ax.plot(data.Forecast)
        ax.plot(forecast_test, color='red')
        st.pyplot(fig)
        
        
        
        st.text("")
        st.subheader("Forecast for the next 12 months", anchor=None)
        
        forecast = pd.DataFrame(model.predict(start=data.index[-1] + 1, end=data.index[-1] + period))
        st.table(forecast.style.background_gradient(cmap='viridis').set_precision(2))


if __name__ == '__main__':
    main()