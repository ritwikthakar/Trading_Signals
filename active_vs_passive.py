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
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1wk'))
st.header(f'{ticker.upper()} Active vs Passive Trading Strategy')


# In[3]:


start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)


# In[4]:


# Active Trading Strategy

# Calculate the Donchian Channels
n = 20
df['Upper Band'] = df['High'].rolling(n).max()
df['Lower Band'] = df['Low'].rolling(n).min()
df['Middle Band'] = (df['Upper Band'] + df['Lower Band']) / 2

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

# Calculate the MACD
df['12EMA'] = df['Close'].ewm(span=12).mean()
df['26EMA'] = df['Close'].ewm(span=26).mean()
df['MACD'] = df['12EMA'] - df['26EMA']
df['Signal Line'] = df['MACD'].ewm(span=9).mean()
df['Histogram'] = df['MACD'] - df['Signal Line']

# Calculate the 9SMA and 20SMA
df['5SMA'] = df['Close'].rolling(window=5).mean()
df['9SMA'] = df['Close'].rolling(window=9).mean()
df['20SMA'] = df['Close'].rolling(window=20).mean()

# Calculate the Bollinger Band Width

def calculate_bollinger_bands(df, window_size, num_std):
    rolling_mean = df['Close'].rolling(window=window_size).mean()
    rolling_std = df['Close'].rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bollinger_width = (upper_band - lower_band) / rolling_mean
    return bollinger_width

window_size = 20
num_std = 2
df["Bollinger_Width"] = calculate_bollinger_bands(df, window_size, num_std)
df["Bollinger_Width_Avg"] = df['Bollinger_Width'].rolling(window=20).mean()

# Calculate the Bollinger Band %B

window = 20
ma = df['Close'].rolling(window).mean()
std = df['Close'].rolling(window).std()
upper_band = ma + 2*std
lower_band = ma - 2*std
bandwidth = (upper_band - lower_band) / ma * 100

def bollinger_band_percent_b(close_prices, window_size=20, num_std_dev=2):
    rolling_mean = close_prices.rolling(window=window_size).mean()
    rolling_std = close_prices.rolling(window=window_size).std()
    upper_band = rolling_mean + num_std_dev * rolling_std
    lower_band = rolling_mean - num_std_dev * rolling_std
    percent_b = (close_prices - lower_band) / (upper_band - lower_band)
    return percent_b

df["Percent_B"] = bollinger_band_percent_b(df['Close'])

# calculate the Average True Range (ATR)
df['tr1'] = abs(df['High'] - df['Low'])
df['tr2'] = abs(df['High'] - df['Close'].shift())
df['tr3'] = abs(df['Low'] - df['Close'].shift())
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['tr'].rolling(n).mean()
df['20atr'] = df['atr'].rolling(window=20).mean()

# calculate the Bollinger Band Trend
def bollinger_band_trend(close_prices, window_size=20, num_std_dev=2):
    rolling_mean = close_prices.rolling(window=window_size).mean()
    rolling_std = close_prices.rolling(window=window_size).std()
    upper_band = rolling_mean + num_std_dev * rolling_std
    lower_band = rolling_mean - num_std_dev * rolling_std
    percent_b = (close_prices - lower_band) / (upper_band - lower_band)
    trend = np.where(percent_b > 0.5, 1, np.where(percent_b < -0.5, -1, 0))
    return pd.Series(trend, index=close_prices.index)

df["BB_Trend"] = bollinger_band_trend(df['Close'])


# In[5]:


# Generate Long Term Trend signals
long = df[(df["Histogram"] > 0) &
                         (df["Histogram"].shift(1) < 0)]
short = df[(df["Histogram"] < 0) &
                          (df["Histogram"].shift(1) > 0)]

# Profit Booking signals to the figure
close_long = df[(df["Percent_B"] > 1) &
                         (df["Percent_B"].shift(1) < 1)]
close_short = df[(df["Percent_B"] < 0) &
                          (df["Percent_B"].shift(1) > 0)]

# Position Reentry signals to the figure
mean_reversion_buy = df[(df["Close"] > df["20SMA"]) &
                         (df["Close"].shift(1) < df["20SMA"].shift(1))]
mean_reversion_sell = df[(df["Close"] < df["20SMA"]) &
                          (df["Close"].shift(1) > df["20SMA"].shift(1))]


# In[21]:


fig = make_subplots(rows=5, cols=1, vertical_spacing = 0.04, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "MACD", "% B", "Bollinger Band Width", "Bollinger Band Trend"))

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

fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], 
                mode='lines', name='Upper Band', line=dict(color='skyblue', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], 
                mode='lines', name='Lower Band', line=dict(color='skyblue', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['Middle Band'], 
                mode='lines', name='Middle Band', line=dict(color='skyblue', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['20SMA'], name='20 SMA', line=dict(color='Orange', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['5SMA'], name='5 SMA', line=dict(color='purple', width=2)))

fig.add_trace(go.Scatter(x=df.index, y=df['9SMA'], name='9 SMA', line=dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=long.index,
                         y=df["Close"],
                         mode="markers",
                         marker=dict(color="green", size=10, symbol='triangle-up'),
                         name="Buy Signal"))

fig.add_trace(go.Scatter(x=short.index,
                         y=df["Close"],
                         mode="markers",
                         marker=dict(color="red", size=10, symbol='triangle-down'),
                         name="Sell Signal"))

fig.add_trace(go.Scatter(x=close_long.index,
                         y=df["High"],
                         mode="markers",
                         marker=dict(color="red", size=10),
                         name="Close Long"))

fig.add_trace(go.Scatter(x=close_short.index,
                         y=df["Low"],
                         mode="markers",
                         marker=dict(color="green", size=10),
                         name="Close Short"))

fig.add_trace(go.Scatter(x=mean_reversion_buy.index,
                         y=df["High"],
                         mode="markers",
                         marker=dict(color="green", size=7, symbol='star'),
                         name="Mean Reversion Buy Signal"))

fig.add_trace(go.Scatter(x=short.index,
                         y=df["Low"],
                         mode="markers",
                         marker=dict(color="red", size=7, symbol='star'),
                         name="Mean Reversion Sell Signal"))

fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=2)), row = 2, col = 1)

fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal', line=dict(color='red', width=2)), row = 2, col = 1)

fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker=dict(color=df['Histogram'], colorscale='rdylgn')), row = 2, col = 1)

fig.add_trace(go.Scatter(x=df.index, y=df['Percent_B'], name='% B', line=dict(color='brown', width=2)), row = 3, col = 1)

fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Width'], name='Bollinger Band Width', line=dict(color='purple', width=2)), row = 4, col = 1)

fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Width_Avg'], name='Mean Bollinger Band Width', line=dict(color='orange', width=2)), row = 4, col = 1)

fig.add_trace(go.Scatter(x=df.index, y=df['BB_Trend'], name='Bollinger Band Trend', line=dict(color='Orange', width=2)), row = 5, col = 1)

# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=1400, width=1200)

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


# In[7]:


# Back Testing Active Trading Strategy 
df['signal'] = np.where(df['MACD'] > df['Signal Line'], 1, 0)
df['signal'] = np.where(df['MACD'] < df['Signal Line'] , -1, df['signal'])
# df['signal'] = np.where(df['MACD'] < df['Signal Line'], 1, 0)
# df['signal'] = np.where(df['Percent_B'] < 0, -1, df['signal'])
df['Return'] = df['Adj Close'].pct_change()
df['System_Return'] = df['signal'] * df['Return']
df['Entry'] = df.signal.diff()
df.dropna(inplace=True)


# In[8]:


# Results of Active Trading Strategy
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df['Return'], 
                mode='lines', name='Buy & Hold Strategy', line=dict(color='blue', width=2)))
fig2.add_trace(go.Scatter(x=df.index, y=df['System_Return'], 
                mode='lines', name='Active Trading Strategy', line=dict(color='Orange', width=2)))


# In[18]:


# Passive Trading Strategy

df['50SMA'] = df['Close'].rolling(window=50).mean()

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

    
# Super Trend

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
atr_multiplier = 3.0


supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)


# In[24]:


fig3 = make_subplots(rows=3, cols=1, vertical_spacing = 0.04, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "RSI", "ATR"))

fig3.append_trace(
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

fig3.add_trace(go.Scatter(x=df.index, y=df['20SMA'], name='20 SMA', line=dict(color='Orange', width=2)))

fig3.add_trace(go.Scatter(x=dates, y=df["psarbull"], name='buy',mode = 'markers',
                         marker = dict(color='green', size=2)))

fig3.add_trace(go.Scatter(x=dates, y=df["psarbear"], name='sell', mode = 'markers',
                         marker = dict(color='red', size=2)))

fig3.add_trace(go.Scatter(x=df.index, y=df['Final Lowerband'], name='Supertrend Lower Band',
                         line = dict(color='green', width=2)))

fig3.add_trace(go.Scatter(x=df.index, y=df['Final Upperband'], name='Supertrend Upper Band',
                         line = dict(color='red', width=2)))

fig3.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='green', width=2)), row = 2, col = 1)

fig3.add_trace(go.Scatter(x=df.index, y=df['20RSI'], name='Mean RSI', line=dict(color='Orange', width=2)), row = 2, col = 1)

fig3.add_trace(go.Scatter(x=df.index, y=df['atr'], name='ATR', line=dict(color='purple', width=2)), row = 3, col = 1)

fig3.add_trace(go.Scatter(x=df.index, y=df['20atr'], name='Mean ATR', line=dict(color='orange', width=2)), row = 3, col = 1)

# Make it pretty
layout_1 = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=1000, width=1200)

if i == '1d':
    fig3.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )
elif i == '1wk':
    fig3.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )
else:
    fig3.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # NOTE: Below values are bound (not single values), ie. hide x to y
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                    # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                ]
                    )

# Update options and show plot
fig3.update_layout(layout_1)


# In[11]:


df['Action'] = np.where(df['Close'] > df['20SMA'] , 1, 0) 
df['Action'] = np.where(df['Close'] < df['20SMA'], -1, df['Action'])
df['PSAR_Action'] = np.where(df['psarbull'] , 1, 0) 
df['PSAR_Action'] = np.where(df['psarbear'] , -1, df['PSAR_Action'])


# In[12]:


def signal(df):
    if df['Action'] == 1 and df['PSAR_Action'] == 1:
        return 1
    elif df['Action'] == -1 and df['PSAR_Action'] == -1:
        return -1
    else:
        return 0


# In[13]:


df['PSAR_Returns'] = df.apply(signal, axis = 1)


# In[16]:


# Back Passive Trading Strategy 
# df['signal'] = np.where(df['MACD'] > df['Signal Line'], 1, 0)
# df['signal'] = np.where(df['MACD'] < df['Signal Line'] , -1, df['signal'])
# df['signal'] = np.where(df['MACD'] < df['Signal Line'], 1, 0)
# df['signal'] = np.where(df['Percent_B'] < 0, -1, df['signal'])
# df['Return'] = df['Adj Close'].pct_change()
df['Passive_Return'] = df['PSAR_Action'] * df['Return']
# df['Passive_Entry'] = df.signal.diff()
# df.dropna(inplace=True)


# In[17]:


# Results of Active Trading Strategy
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df.index, y=df['Return'], 
                mode='lines', name='Buy & Hold Strategy', line=dict(color='blue', width=2)))
fig4.add_trace(go.Scatter(x=df.index, y=df['Passive_Return'], 
                mode='lines', name='Passive Trading Strategy', line=dict(color='Orange', width=2)))


# In[ ]:


tab1, tab2, tab3, tab4 = st.tabs(["Active Trading Strtegy" , "Results of Active Trading Strategy", "Passive Trading Strtegy" , "Results of Passive Trading Strategy"])
    
with tab1:
    st.header("Active Trading Strategy")
    st.plotly_chart(fig)
    
with tab2:
    st.header("Results of Active Trading Strategy Vs Buy & Hold")
    st.plotly_chart(fig2)

with tab3:
    st.header("Passive Trading Strategy")
    st.plotly_chart(fig3)
    
with tab4:
    st.header("Results of Active Trading Strategy Vs Buy & Hold")
    st.plotly_chart(fig4)
