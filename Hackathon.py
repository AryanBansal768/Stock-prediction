import streamlit as st
import yfinance as fin
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.graph_objects as go
import mysql.connector




st.header("Stock Prediction")
def get_ticker(name):
    com= fin.Ticker(name)
    return com

st.sidebar.subheader("Parameters")


start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))
infoo=["Data","Closing price prediction","Low price prediction","Open price prediction"]
information=st.sidebar.selectbox("Information Needed",infoo)

ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
ticker_name=st.sidebar.selectbox('Stock Ticker', ticker_list)
c= get_ticker(ticker_name)


ticker_logo= fin.Ticker(ticker_name)
string_logo = '<img src=%s>' % ticker_logo.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)
data= fin.download(ticker_name, start=start_date,end=end_date)
data.reset_index(inplace=True)

if(information=="Data"):

    #st.header(c.info['longName'])
    #st.write(c.info['longBusinessSummary'])
    st.subheader("Ticker Data:")
    st.write(data)
    st.subheader("Data chart for Open and Close Price:")
    fig =plt.figure()
    
    plt.plot(data.Open)
    plt.plot(data.Close)
    plt.legend("Open_values")
    st.pyplot(fig)

if(information=="Closing price prediction"):
    st.header(c.info['longName'])
    st.write(c.info['longBusinessSummary'])
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Close)
    st.subheader("Closing Price Vs Time Chart:")
    st.pyplot(fig)


    ma100=data.Close.rolling(100).mean()
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Close)
    plt.plot(ma100, 'r')
    plt.legend("Cm")
    st.subheader("Closing Price Vs Time Chart with 100ma:")
    st.pyplot(fig)


    ma100=data.Close.rolling(100).mean()
    ma200=data.Close.rolling(200).mean()
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Close)
    plt.plot(ma100, 'r')
    plt.plot(ma200,'y')
    plt.legend("CmM")
    st.subheader("Closing Price Vs Time Chart with 100ma and 200ma:")
    st.pyplot(fig)


    x = data[['Open', 'High','Low', 'Volume']]
    y = data['Close']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15 , shuffle=False,random_state = 0)

    regression = LinearRegression()
    regression.fit(train_x, train_y)
    regression_confidence = regression.score(test_x, test_y)
    predicted=regression.predict(test_x)
    dfr=pd.DataFrame({'Actual_Price':test_y, 'Predicted_Price':predicted})

    x2 = dfr.Actual_Price.mean()
    y2 = dfr.Predicted_Price.mean()
    Accuracy1 = x2/y2*100
    st.write(dfr.Predicted_Price,dfr.Actual_Price)

    fig=plt.figure(figsize= (12,6))
    plt.title("Predictied Vs Actual Values")
    plt.plot(dfr.Actual_Price, color='black')
    plt.plot(dfr.Predicted_Price, color='red')
    plt.legend("AP")
    st.pyplot(fig)





if(information=="Low price prediction"):
    st.header(c.info['longName'])
    st.write(c.info['longBusinessSummary'])
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Low)
    st.subheader("Low Price Vs Time Chart:")
    st.pyplot(fig)


    ma100=data.Low.rolling(100).mean()
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Low)
    plt.plot(ma100, 'r')
    st.subheader("Low Price Vs Time Chart with 100ma:")
    st.pyplot(fig)


    ma100=data.Low.rolling(100).mean()
    ma200=data.Low.rolling(200).mean()
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Low)
    plt.plot(ma100, 'r')
    plt.plot(ma200,'y')
    st.subheader("Low Price Vs Time Chart with 100ma and 200ma:")
    st.pyplot(fig)


    x = data[['Open', 'High','Close', 'Volume']]
    y = data['Low']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15 , shuffle=False,random_state = 0)

    regression = LinearRegression()
    regression.fit(train_x, train_y)
    regression_confidence = regression.score(test_x, test_y)
    predicted=regression.predict(test_x)
    dfr=pd.DataFrame({'Actual_Price':test_y, 'Predicted_Price':predicted})

    x2 = dfr.Actual_Price.mean()
    y2 = dfr.Predicted_Price.mean()
    Accuracy1 = x2/y2*100
    st.write(dfr.Predicted_Price,dfr.Actual_Price)

    
    
    fig=plt.figure(figsize= (12,6))
    plt.title("Predictied Vs Actual Values")
    plt.plot(dfr.Actual_Price, color='black')
    plt.plot(dfr.Predicted_Price, color='red')
    st.pyplot(fig)


if(information=="Open price prediction"):
    st.header(c.info['longName'])
    st.write(c.info['longBusinessSummary'])
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Open)
    st.subheader("Open Price Vs Time Chart:")
    st.pyplot(fig)


    ma100=data.Open.rolling(100).mean()
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Open)
    plt.plot(ma100, 'r')
    st.subheader("Open Price Vs Time Chart with 100ma:")
    st.pyplot(fig)


    ma100=data.Open.rolling(100).mean()
    ma200=data.Open.rolling(200).mean()
    fig=plt.figure(figsize= (12,6))
    plt.plot(data.Open)
    plt.plot(ma100, 'r')
    plt.plot(ma200,'y')
    st.subheader("Open Price Vs Time Chart with 100ma and 200ma:")
    st.pyplot(fig)


    x = data[['Low', 'High','Close', 'Volume']]
    y = data['Open']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15 , shuffle=False,random_state = 0)

    regression = LinearRegression()
    regression.fit(train_x, train_y)
    regression_confidence = regression.score(test_x, test_y)
    predicted=regression.predict(test_x)
    dfr=pd.DataFrame({'Actual_Price':test_y, 'Predicted_Price':predicted})

    x2 = dfr.Actual_Price.mean()
    y2 = dfr.Predicted_Price.mean()
    Accuracy1 = x2/y2*100
    st.write(dfr.Predicted_Price,dfr.Actual_Price)

    
    
    fig=plt.figure(figsize= (12,6))
    plt.title("Predictied Vs Actual Values")
    plt.plot(dfr.Actual_Price, color='black')
    plt.plot(dfr.Predicted_Price, color='red')
    st.pyplot(fig)