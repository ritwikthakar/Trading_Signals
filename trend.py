# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:48:27 2023

@author: ritwi
"""

from datetime import timedelta
import datetime as dt
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

df = pd.DataFrame()

ticker = st.sidebar.text_input('Enter Ticker', 'SPY')
# t = st.sidebar.selectbox('Select Number of Days', ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'))
# i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'))
t = st.sidebar.selectbox('Select Number of Days', (180, 3000, 1000, 735, 450, 400, 350, 252, 150, 90, 60, 45, 30, 15))
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1wk', '1h', '15m'))
st.header(f'{ticker.upper()} Technical Indicators')

start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)

# Heikin Ashi
df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

# Calculate the RSI
n = 14
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(n).mean()
avg_loss = loss.rolling(n).mean().abs()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
df['20RSI'] = df['RSI'].rolling(window=20).mean()

# Copy df for Double Supertrend
df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df4 = df.copy()

# Compute RSI divergence
def compute_rsi_divergence(data, window):
    high = df["High"].rolling(window).max()
    low = df["Low"].rolling(window).min()
    rsi = df["RSI"]
    divergence = (rsi - rsi.shift(window)) / (high - low)
    return divergence

rsi_divergence_window = 10
df["RSI_Divergence"] = compute_rsi_divergence(df, rsi_divergence_window)

# Compute buy and sell signals
buy_signal = (df["RSI_Divergence"] > 0) & (df["RSI_Divergence"].shift(1) < 0)
sell_signal = (df["RSI_Divergence"] < 0) & (df["RSI_Divergence"].shift(1) > 0)

# Calculate the MACD
df['12EMA'] = df['Close'].ewm(span=12).mean()
df['26EMA'] = df['Close'].ewm(span=26).mean()
df['MACD'] = df['12EMA'] - df['26EMA']
df['Signal Line'] = df['MACD'].ewm(span=9).mean()
df['Histogram'] = df['MACD'] - df['Signal Line']

# calculate the Average True Range (ATR)
df['tr1'] = abs(df['High'] - df['Low'])
df['tr2'] = abs(df['High'] - df['Close'].shift())
df['tr3'] = abs(df['Low'] - df['Close'].shift())
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['tr'].rolling(n).mean()

# calculate the Average Directional Index (ADX)
df['up_move'] = df['High'] - df['High'].shift()
df['down_move'] = df['Low'].shift() - df['Low']
df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
df['plus_di'] = 100 * (df['plus_dm'] / df['atr']).ewm(span=n, adjust=False).mean()
df['minus_di'] = 100 * (df['minus_dm'] / df['atr']).ewm(span=n, adjust=False).mean()
df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])).ewm(span=n, adjust=False).mean()
df['adx'] = df['dx'].ewm(span=n, adjust=False).mean()

# Calculate PSAR
def psar(df, iaf = 0.02, maxaf = 0.2):
    length = len(df)
    dates = list(df.index)
    high = list(df['High'])
    low = list(df['Low'])
    close = list(df['Close'])
    psar = close[0:len(close)]
    psarbull = [None] * length # Bullish signal - dot below candle
    psarbear = [None] * length # Bearish signal - dot above candle
    bull = True
    af = iaf # acceleration factor
    ep = low[0] # ep = Extreme Point
    hp = high[0] # High Point
    lp = low[0] # Low Point

    # https://www.investopedia.com/terms/p/parabolicindicator.asp - Parabolic Stop & Reverse Formula from Investopedia 
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}

if __name__ == "__main__":
    import sys
    import os
    
    startidx = 0
    endidx = len(df)
    
    result = psar(df)
    dates = result['dates'][startidx:endidx]
    close = result['close'][startidx:endidx]
    df["psarbear"] = result['psarbear'][startidx:endidx]
    df["psarbull"] = result['psarbull'][startidx:endidx]

# Supertrend

def Supertrend(df, atr_period, multiplier):
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # calculate ATR
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    # df['atr'] = df['tr'].rolling(atr_period).mean()
    
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    
    # initialize Supertrend column to True
    supertrend = [True] * len(df)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
        
        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            
            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan
    
    return pd.DataFrame({
        'Supertrend': supertrend,
        'Final Lowerband': final_lowerband,
        'Final Upperband': final_upperband
    }, index=df.index)
    
    
atr_period = 10
atr_multiplier = 3


supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)

# Fast Double Supertrend
st_1 = Supertrend(df1, 14, 2)
df1 = df1.join(st_1)
st_2 = Supertrend(df2, 21, 1)
df2 = df2.join(st_2)

# Slow Double Supertrend
st_3 = Supertrend(df3, 21, 3)
df3 = df3.join(st_3)
st_4 = Supertrend(df4, 20, 7)
df4 = df4.join(st_4)

def create_plot(df, indicators):
    fig = sp.make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.15, 0.15, 0.15], vertical_spacing=0.02, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "Lower Indicator 1", "Lower Indicator 2", "Lower Indicator 3", "Lower Indicator 4"))

    for indicator in indicators:
        if indicator == 'Candlestick Chart':
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        elif indicator == 'Heikin Ashi Candles':
            fig.add_trace(go.Candlestick(x=df.index, open=df["HA_Open"], high=df["HA_High"], low=df["HA_Low"], close=df["HA_Close"], name="Price"), row=1, col=1)
        elif indicator == 'RSI':
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index[buy_signal], y=df["RSI"][buy_signal], mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"), name="Buy"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index[sell_signal], y=df["RSI"][sell_signal], mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"), name="Sell"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['20RSI'], name='Mean RSI', line=dict(color='Orange', width=2)), row = 2, col = 1)
        elif indicator == 'MACD':
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=2)), row = 3, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal', line=dict(color='red', width=2)), row = 3, col = 1)
            fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker=dict(color=df['Histogram'], colorscale='rdylgn')), row = 3, col = 1)
        elif indicator == 'ATR':
            fig.add_trace(go.Scatter(x=df.index, y=df['atr'], name='ATR', line=dict(color='purple', width=2)), row = 4, col = 1)
        elif indicator == 'ADX':
            fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX', line=dict(color='blue', width=2)), row = 5, col = 1)
        elif indicator == 'PSAR':
            fig.add_trace(go.Scatter(x=dates, y=df["psarbull"], name='buy',mode = 'markers', marker = dict(color='green', size=2)))
            fig.add_trace(go.Scatter(x=dates, y=df["psarbear"], name='sell', mode = 'markers',marker = dict(color='red', size=2)))
        elif indicator == 'Supertrend':
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Lowerband'], name='Supertrend Lower Band', line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Upperband'], name='Supertrend Upper Band', line = dict(color='red', width=2)))
        elif indicator == 'Fast Double Supertrend':
            fig.add_trace(go.Scatter(x=df1.index, y=df1['Final Lowerband'], name='Supertrend Fast Lower Band', line = dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=df1.index, y=df1['Final Upperband'], name='Supertrend Fast Upper Band', line = dict(color='purple', width=2)))
            fig.add_trace(go.Scatter(x=df2.index, y=df2['Final Lowerband'], name='Supertrend Slow Lower Band',line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df2.index, y=df2['Final Upperband'], name='Supertrend Slow Upper Band',line = dict(color='red', width=2)))
        elif indicator == 'Slow Double Supertrend':
            fig.add_trace(go.Scatter(x=df3.index, y=df3['Final Lowerband'], name='Supertrend Fast Lower Band', line = dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=df3.index, y=df3['Final Upperband'], name='Supertrend Fast Upper Band', line = dict(color='purple', width=2)))
            fig.add_trace(go.Scatter(x=df4.index, y=df4['Final Lowerband'], name='Supertrend Slow Lower Band',line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df4.index, y=df4['Final Upperband'], name='Supertrend Slow Upper Band',line = dict(color='red', width=2)))
    
    # Make it pretty
    layout = go.Layout(
        plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=20,
        height=1000, width=1200)

    if i == '1d':
        fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                        # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                    ]
                        )
    elif i == '1wk':
        fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                        # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                    ]
                        )
    else:
        fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                        # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                    ]
                        )
    
    fig.update_layout(layout)
    st.plotly_chart(fig)


indicators = ['Candlestick Chart', 'Heikin Ashi Candles', 'RSI', 'MACD', 'ATR', 'ADX', 'PSAR', 'Supertrend', 'Fast Double Supertrend', 'Slow Double Supertrend']

default_options = ['Candlestick Chart', 'RSI', 'MACD', 'ATR', 'ADX', 'PSAR', 'Supertrend']


selected_indicators = st.multiselect('Select Indicators', indicators, default = default_options)


create_plot(df, selected_indicators)