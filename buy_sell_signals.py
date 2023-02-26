#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import streamlit as st


# In[2]:


ticker = st.sidebar.text_input('Enter Ticker', 'SPY')
# t = st.sidebar.selectbox('Select Number of Days', ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'))
# i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'))
t = st.sidebar.selectbox('Select Number of Days', (180, 3000, 1000, 735, 400, 350, 252, 150, 90, 60, 45, 30, 15))
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1wk', '1h', '15m'))
st.header(f'{ticker.upper()} Technical Indicators')


# In[3]:


start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)
# df = yf.download(ticker, period=t, interval= i)


# In[ ]:


def calculate_bollinger_bands(df, window_size, num_std):
    rolling_mean = df['Close'].rolling(window=window_size).mean()
    rolling_std = df['Close'].rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bollinger_width = (upper_band - lower_band) / rolling_mean
    return bollinger_width


# In[ ]:


window_size = 20
num_std = 2
bollinger_width = calculate_bollinger_bands(df, window_size, num_std)


# In[ ]:


window = 20
ma = df['Close'].rolling(window).mean()
std = df['Close'].rolling(window).std()
upper_band = ma + 2*std
lower_band = ma - 2*std
bandwidth = (upper_band - lower_band) / ma * 100


# In[ ]:


def bollinger_band_percent_b(close_prices, window_size=20, num_std_dev=2):
    rolling_mean = close_prices.rolling(window=window_size).mean()
    rolling_std = close_prices.rolling(window=window_size).std()
    upper_band = rolling_mean + num_std_dev * rolling_std
    lower_band = rolling_mean - num_std_dev * rolling_std
    percent_b = (close_prices - lower_band) / (upper_band - lower_band)
    return percent_b


# In[ ]:


percent_b = bollinger_band_percent_b(df['Close'])


# In[ ]:


def bollinger_band_trend(close_prices, window_size=20, num_std_dev=2):
    rolling_mean = close_prices.rolling(window=window_size).mean()
    rolling_std = close_prices.rolling(window=window_size).std()
    upper_band = rolling_mean + num_std_dev * rolling_std
    lower_band = rolling_mean - num_std_dev * rolling_std
    percent_b = (close_prices - lower_band) / (upper_band - lower_band)
    trend = np.where(percent_b > 0.5, 1, np.where(percent_b < -0.5, -1, 0))
    return pd.Series(trend, index=close_prices.index)


# In[ ]:


trend = bollinger_band_trend(df['Close'])


# In[ ]:


def sma(close_prices, window_size=20):
    sma = close_prices.rolling(window=window_size).mean()
    return sma


# In[ ]:


sma_5 = sma(df['Close'], window_size=5)
sma_9 = sma(df['Close'], window_size=9)
sma_50 = sma(df['Close'], window_size=50)
sma_200 = sma(df['Close'], window_size=200)


# In[ ]:


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


# In[ ]:


if __name__ == "__main__":
    import sys
    import os
    
    startidx = 0
    endidx = len(df)
    
    result = psar(df)
    dates = result['dates'][startidx:endidx]
    close = result['close'][startidx:endidx]
    psarbear = result['psarbear'][startidx:endidx]
    psarbull = result['psarbull'][startidx:endidx]


# In[ ]:


# Define the periods for the SMAs
short_period = 9
long_period = 20

# Calculate the short and long SMAs
df["SMA_short"] = df["Close"].rolling(window=short_period).mean()
df["SMA_long"] = df["Close"].rolling(window=long_period).mean()


# In[ ]:


# Add the buy and sell signals to the figure
buy_signals = df[(df["SMA_short"] > df["SMA_long"]) &
                         (df["SMA_short"].shift(1) < df["SMA_long"].shift(1))]
sell_signals = df[(df["SMA_short"] < df["SMA_long"]) &
                          (df["SMA_short"].shift(1) > df["SMA_long"].shift(1))]


# In[ ]:


fig = make_subplots(rows=4, cols=1, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "Bollinger Bands Width", 'Bollinger band %B', 'Bollinger Band Trend'))

fig.append_trace(
    go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Adj Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    ), row=1, col=1
)

fig.add_trace(go.Scatter(x=dates, y=psarbull, name='buy',mode = 'markers',
                         marker = dict(color='green', size=2)))

fig.add_trace(go.Scatter(x=dates, y=psarbear, name='sell', mode = 'markers',
                         marker = dict(color='red', size=2)))

fig.add_trace(go.Scatter(x=df.index, y=upper_band, name='Upper Bollinger Band', line=dict(color='black', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=lower_band, name='Lower Bollinger Band', line=dict(color='black', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=ma, name='20 SMA', line=dict(color='Orange', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=sma_5, name='5 SMA', line=dict(color='purple', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=sma_9, name='9 SMA', line=dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=sma_50, name='50 SMA', line=dict(color='green', width=2)))

fig.add_trace(go.Scatter(x=buy_signals.index,
                         y=df["Close"],
                         mode="markers",
                         marker=dict(color="green", size=10),
                         name="Buy Signal"))

fig.add_trace(go.Scatter(x=sell_signals.index,
                         y=df["Close"],
                         mode="markers",
                         marker=dict(color="red", size=10),
                         name="Sell Signal"))

fig.add_trace(go.Scatter(x=df.index, y=bandwidth, name='Bandwidth'), row = 2, col = 1)

fig.add_trace(go.Scatter(x=df.index, y=percent_b, name='% B'), row = 3, col = 1)

fig.add_trace(go.Scatter(x=df.index, y=trend, name='Bollinger Band Trend'), row = 4, col = 1)

# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=1200, width=1000)

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

# Update options and show plot
fig.update_layout(layout)


# In[5]:



# Define MACD function
def MACD(data, n_fast, n_slow, n_signal):
    EMAfast = data.ewm(span=n_fast, min_periods=n_slow).mean()
    EMAslow = data.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = EMAfast - EMAslow
    signal = MACD.ewm(span=n_signal, min_periods=n_slow).mean()
    histogram = MACD - signal
    return MACD, signal, histogram


close = df['Close']

# Calculate MACD
MACD, signal, histogram = MACD(close, 12, 26, 9)

# Generate signals
signals = []
for i in range(len(close)):
    if i == 0:
        signals.append(None)
    elif MACD[i] > signal[i] and MACD[i-1] <= signal[i-1]:
        signals.append('Buy')
    elif MACD[i] < signal[i] and MACD[i-1] >= signal[i-1]:
        signals.append('Sell')
    else:
        signals.append(None)

# Plot results
fig1 = make_subplots(rows=2, cols=1, vertical_spacing = 0.05, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "MACD"))

fig1.append_trace(
    go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Adj Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False, 
    ), row=1, col=1
)

fig1.add_trace(go.Scatter(x=dates, y=psarbull, name='buy',mode = 'markers',
                         marker = dict(color='green', size=2)))

fig1.add_trace(go.Scatter(x=dates, y=psarbear, name='sell', mode = 'markers',
                         marker = dict(color='red', size=2)))


fig1.add_trace(go.Scatter(x=df.index, y=MACD, name='MACD', line=dict(color='blue', width=2)), row = 2, col = 1)

fig1.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='red', width=2)), row = 2, col = 1)

fig1.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram', marker=dict(color=histogram, colorscale='rdylgn')), row = 2, col = 1)

fig1.add_trace(
    go.Scatter(
        x=df.index,
        y=[None if signal is None else df['Close'].min() for signal in signals],
        name='Buy',
        mode='markers',
        marker=dict(color='green', symbol='triangle-up')
    )
)

fig1.add_trace(
    go.Scatter(
        x=df.index,
        y=[None if signal is None else df['Close'].max() for signal in signals],
        name='Sell',
        mode='markers',
        marker=dict(color='red', symbol='triangle-down')
    )
)

fig1.update_xaxes(rangeslider_visible=False)

layout_1 = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=600, width=800)

fig1.update_layout(layout_1)


# In[ ]:


# Define RSI function
def RSI(data, n):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI


close = df['Close']

# Calculate RSI
rsi = RSI(close, 14)

# Calculate divergence
highs = close.rolling(14).max()
lows = close.rolling(14).min()
divergence = ((rsi - 50) * (highs - lows) / (rsi + 50)).rolling(14).sum()

# Generate signals
signals = []
for i in range(len(close)):
    if i == 0:
        signals.append(None)
    elif divergence[i] > divergence[i-1] and rsi[i] > 30 and rsi[i-1] <= 30:
        signals.append('Buy')
    elif divergence[i] < divergence[i-1] and rsi[i] < 70 and rsi[i-1] >= 70:
        signals.append('Sell')
    else:
        signals.append(None)

# Plot results
# fig2 = go.Figure()

fig2 = make_subplots(rows=2, cols=1, vertical_spacing = 0.05, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "RSI"))

fig2.append_trace(
    go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Adj Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False, 
    ), row=1, col=1
)

fig2.add_trace(go.Scatter(x=dates, y=psarbull, name='buy',mode = 'markers',
                         marker = dict(color='green', size=2)))

fig2.add_trace(go.Scatter(x=dates, y=psarbear, name='sell', mode = 'markers',
                         marker = dict(color='red', size=2)))


fig2.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='blue', width=2)), row = 2, col = 1)

# fig2.add_trace(
#     go.Scatter(
#         x=df.index,
#         y=df['Close'],
#         name='Price',
#         mode='lines'
#     )
# )

# fig2.add_trace(
#     go.Scatter(
#         x=df.index,
#         y=rsi,
#         name='RSI',
#         mode='lines'
#     )
# )

fig2.add_trace(
    go.Scatter(
        x=df.index,
        y=[None if signal is None else df['Close'].min() for signal in signals],
        name='Buy',
        mode='markers',
        marker=dict(color='green', symbol='triangle-up')
    )
)

fig2.add_trace(
    go.Scatter(
        x=df.index,
        y=[None if signal is None else df['Close'].max() for signal in signals],
        name='Sell',
        mode='markers',
        marker=dict(color='red', symbol='triangle-down')
    )
)

fig2.update_xaxes(rangeslider_visible=False)

fig2.update_layout(layout_1)


# In[6]:


df.reset_index(inplace=True)

# Define Donchian channel function
def donchian_channel(data, n):
    high = data['High'].rolling(n).max()
    low = data['Low'].rolling(n).min()
    mid = (high + low) / 2
    upper_band = high.shift(1)
    lower_band = low.shift(1)
    return upper_band, mid, lower_band

# Calculate Donchian channels
upper_band, mid, lower_band = donchian_channel(df, 20)

# Generate signals
signals = []
for i in range(len(df)):
    if df['Close'][i] > upper_band[i] and df['Close'][i-1] <= upper_band[i-1]:
        signals.append('Buy')
    elif df['Close'][i] < lower_band[i] and df['Close'][i-1] >= lower_band[i-1]:
        signals.append('Sell')
    else:
        signals.append(None)

# Plot results
fig3 = make_subplots(rows=2, cols=1,vertical_spacing = 0.05, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart Donchain Channels", "Volume"))

# Add candlestick chart
fig3.add_trace(
    go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ),
    row=1, col=1
)

# Add Donchian channels
fig3.add_trace(
    go.Scatter(
        x=df['Date'],
        y=upper_band,
        name='Upper Band',
        mode='lines',
        line=dict(color='purple')
    ),
    row=1, col=1
)

fig3.add_trace(
    go.Scatter(
        x=df['Date'],
        y=mid,
        name='Mid',
        mode='lines',
        line=dict(color='black')
    ),
    row=1, col=1
)

fig3.add_trace(
    go.Scatter(
        x=df['Date'],
        y=lower_band,
        name='Lower Band',
        mode='lines',
        line=dict(color='purple')
    ),
    row=1, col=1
)

fig3.add_trace(go.Scatter(x=dates, y=psarbull, name='buy',mode = 'markers',
                         marker = dict(color='green', size=2)))

fig3.add_trace(go.Scatter(x=dates, y=psarbear, name='sell', mode = 'markers',
                         marker = dict(color='red', size=2)))


# Add buy and sell signals
fig3.add_trace(
    go.Scatter(
        x=df['Date'],
        y=[None if signal is None else df['Low'].min() for signal in signals],
        name='Buy',
        mode='markers',
        marker=dict(color='green', symbol='triangle-up')
    ),
    row=1, col=1
)

fig3.add_trace(
    go.Scatter(
        x=df['Date'],
        y=[None if signal is None else df['High'].max() for signal in signals],
        name='Sell',
        mode='markers',
        marker=dict(color='red', symbol='triangle-down')
    ),
    row=1, col=1
)

# Add volume chart
fig3.add_trace(
    go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker=dict(color='blue')
    ),
    row=2, col=1
)

fig3.update_xaxes(rangeslider_visible=False)

fig3.update_layout(layout_1)


# In[ ]:


tab1, tab2, tab3, tab4 = st.tabs(['Bollinger Bands' , "MACD","RSI", "Donchian Channels"])
    
with tab1:
    st.header("Bollinger Bands")
    st.plotly_chart(fig)
    
with tab2:
    st.header("MACD")
    st.plotly_chart(fig1)

with tab3:
    st.header("RSI")
    st.plotly_chart(fig2)

with tab4:
    st.header("Donchian Channels")
    st.plotly_chart(fig3)

