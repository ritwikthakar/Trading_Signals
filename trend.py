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

ticker = st.sidebar.text_input('Enter Ticker', 'AAPL')
# t = st.sidebar.selectbox('Select Number of Days', ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'))
# i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'))
t = st.sidebar.selectbox('Select Number of Days', (180, 3000, 1000, 735, 450, 400, 350, 252, 150, 90, 60, 45, 30, 15))
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1wk', '1h', '15m'))
st.header(f'{ticker.upper()} Technical & Fundamental Analysis')

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

# # Compute RSI divergence
# def compute_rsi_divergence(data, window):
#     high = df["High"].rolling(window).max()
#     low = df["Low"].rolling(window).min()
#     rsi = df["RSI"]
#     divergence = (rsi - rsi.shift(window)) / (high - low)
#     return divergence

# rsi_divergence_window = 10
# df["RSI_Divergence"] = compute_rsi_divergence(df, rsi_divergence_window)

# Compute buy and sell signals
# buy_signal = (df["RSI_Divergence"] > 0) & (df["RSI_Divergence"].shift(1) < 0)
# sell_signal = (df["RSI_Divergence"] < 0) & (df["RSI_Divergence"].shift(1) > 0)

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
    
    
atr_period = 7
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

# Engulfing Candles
def find_engulfing_candles(prices):
    # Compute the candle body size for each candle
    prices['BodySize'] = abs(prices['Open'] - prices['Close'])
    
    # Check for bullish engulfing patterns
    bullish_engulfing = (prices['Open'] > prices['Close'].shift(1)) & \
                        (prices['Close'] < prices['Open'].shift(1)) & \
                        (prices['BodySize'] > prices['BodySize'].shift(1))
    
    # Check for bearish engulfing patterns
    bearish_engulfing = (prices['Open'] < prices['Close'].shift(1)) & \
                        (prices['Close'] > prices['Open'].shift(1)) & \
                        (prices['BodySize'] > prices['BodySize'].shift(1))
    
    # Create a new DataFrame to store engulfing candles
    engulfing_candles = pd.DataFrame(index=prices.index)
    engulfing_candles['Bullish'] = bullish_engulfing
    engulfing_candles['Bearish'] = bearish_engulfing
    
    return engulfing_candles
engulfing_candles = find_engulfing_candles(df)

# Fractals
def find_fractals(data):
    fractals = []
    for i in range(9, len(df) - 9):
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1] and \
           df['High'][i] > df['High'][i-2] and df['High'][i] > df['High'][i+2]:
            fractals.append((df.index[i], df['High'][i], 'peak'))
        elif df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1] and \
             df['Low'][i] < df['Low'][i-2] and df['Low'][i] < df['Low'][i+2]:
            fractals.append((df.index[i], df['Low'][i], 'trough'))
    return fractals

fractals = find_fractals(df)

# Define the function to calculate Ichimoku Cloud components
nine_period_high = df['High'].rolling(window=9).max()
nine_period_low = df['Low'].rolling(window=9).min()
df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
period26_high = df['High'].rolling(window=26).max()
period26_low = df['Low'].rolling(window=26).min()
df['kijun_sen'] = (period26_high + period26_low) / 2
df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
period52_high = df['High'].rolling(window=52).max()
period52_low = df['Low'].rolling(window=52).min()
df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
df['chikou_span'] = df['Close'].shift(-26)

def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'


# Calculate the 9SMA and 20SMA
df['5SMA'] = df['Close'].rolling(window=5).mean()
df['9SMA'] = df['Close'].rolling(window=9).mean()
df['20SMA'] = df['Close'].rolling(window=20).mean()
df['50SMA'] = df['Close'].rolling(window=50).mean()
df['200SMA'] = df['Close'].rolling(window=200).mean()
rolling_std = df['Close'].rolling(window=20).std()
df['upper_band'] = df['20SMA'] + (rolling_std * 2)
df['lower_band'] = df['20SMA'] - (rolling_std * 2)

def create_plot(df, indicators):
    fig = sp.make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.15, 0.15, 0.15], vertical_spacing=0.02, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "Lower Indicator 1", "Lower Indicator 2", "Lower Indicator 3", "Lower Indicator 4"))

    for indicator in indicators:
        if indicator == 'Candlestick Chart':
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        elif indicator == 'Heikin Ashi Candles':
            fig.add_trace(go.Candlestick(x=df.index, open=df["HA_Open"], high=df["HA_High"], low=df["HA_Low"], close=df["HA_Close"], name="Price"), row=1, col=1)
        elif indicator == 'RSI':
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)
            # fig.add_trace(go.Scatter(x=df.index[buy_signal], y=df["RSI"][buy_signal], mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"), name="Buy"), row=2, col=1)
            # fig.add_trace(go.Scatter(x=df.index[sell_signal], y=df["RSI"][sell_signal], mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"), name="Sell"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['20RSI'], name='Mean RSI', line=dict(color='Orange', width=2)), row = 2, col = 1)
        elif indicator == 'MACD':
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=2)), row = 3, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal', line=dict(color='red', width=2)), row = 3, col = 1)
            colors = ['green' if val > 0 else 'red' for val in df['Histogram']]
            fig.add_trace(go.Bar(x=df.index, y= df['Histogram'],  marker_color=colors, showlegend = False), row = 3, col=1)
        elif indicator == 'ATR':
            fig.add_trace(go.Scatter(x=df.index, y=df['atr'], name='ATR', line=dict(color='purple', width=2)), row = 4, col = 1)
        elif indicator == 'ADX':
            fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX', line=dict(color='blue', width=2)), row = 5, col = 1)
        # elif indicator == 'PSAR':
        #     fig.add_trace(go.Scatter(x=dates, y=df["psarbull"], name='buy',mode = 'markers', marker = dict(color='green', size=2)))
        #     fig.add_trace(go.Scatter(x=dates, y=df["psarbear"], name='sell', mode = 'markers',marker = dict(color='red', size=2)))
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
        elif indicator == 'SMA Ribbons':
            fig.add_trace(go.Scatter(x=df.index, y=df['5SMA'], name='5 SMA', line=dict(color='purple', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['9SMA'], name='9 SMA', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['50SMA'], name='50 SMA', line=dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['200SMA'], name='200 SMA', line=dict(color='red', width=2)))
        elif indicator == 'Bollinger Bands':
            fig.add_trace(go.Scatter(x=df.index, y=df['20SMA'], name='20 SMA', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], name='Upper BB', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], name='Lower BB', line=dict(color='black', width=2)))
        elif indicator == "Engulfing Candles":
            bullish_engulfing_dates = engulfing_candles[engulfing_candles['Bullish']].index
            fig.add_trace(go.Scatter(x=bullish_engulfing_dates, y=df.loc[bullish_engulfing_dates, 'Low'], mode='markers', name='Bullish Engulfing', marker=dict(color='red', size=10)))
            # Add bearish engulfing candles
            bearish_engulfing_dates = engulfing_candles[engulfing_candles['Bearish']].index
            fig.add_trace(go.Scatter(x=bearish_engulfing_dates, y=df.loc[bearish_engulfing_dates, 'High'], mode='markers', name='Bearish Engulfing', marker=dict(color='green', size=10)))
        elif indicator == 'Fractals':
            for date, price, marker_type in fractals:
                fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers', marker=dict(color='red' if marker_type == 'peak' else 'green'), name=marker_type))
        elif indicator == 'Ichimoku Cloud':
            fig.add_trace(go.Scatter(x=df.index, y=df['tenkan_sen'], line=dict(color='blue', width=2), name='Tenkan-sen'))
            fig.add_trace(go.Scatter(x=df.index, y=df['kijun_sen'], line=dict(color='purple', width=2), name='Kijun-sen'))
            fig.add_trace(go.Scatter(x=df.index, y=df['chikou_span'], line=dict(color='orange', width=2), name='Chikou Span'))
            fig.add_trace(go.Scatter(x=df.index, y=df['senkou_span_a'], line=dict(color='green', width=2), name='Senkou Span A'))
            fig.add_trace(go.Scatter(x=df.index, y=df['senkou_span_b'], line=dict(color='red', width=2), name='Senkou Span B'))
            df['label'] = np.where(df['senkou_span_a'] > df['senkou_span_b'], 1, 0)
            df['group'] = df['label'].ne(df['label'].shift()).cumsum()
            df = df.groupby('group')
            dfs = []
            for name, data in df:
                dfs.append(data)
            for df in dfs:
                fig.add_traces(go.Scatter(x=df.index, y = df['senkou_span_a'],
                                          line = dict(color='rgba(0,0,0,0)')))
                fig.add_traces(go.Scatter(x=df.index, y = df['senkou_span_b'],
                                          line = dict(color='rgba(0,0,0,0)'), 
                                          fill='tonexty', 
                                          fillcolor = get_fill_color(df['label'].iloc[0])))
            
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

symbol = yf.Ticker(ticker)

def ddm_valuation(dividend, growth_rate, discount_rate):
    """
    Perform Dividend Discount Model (DDM) valuation.
    
    Parameters:
    dividend (float): The most recent annual dividend.
    growth_rate (float): The dividend growth rate.
    discount_rate (float): The discount rate (required rate of return).
    
    Returns:
    float: The intrinsic value of the stock.
    """
    # Gordon Growth Model: Value = Dividend / (Discount Rate - Growth Rate)
    intrinsic_value = dividend / (discount_rate - growth_rate)
    return intrinsic_value

# Discounted Cash Flow (DCF) Valuation Function
def dcf_valuation(free_cash_flow, growth_rate, discount_rate, terminal_growth_rate, years=5):
    """
    Perform DCF valuation.
    
    Parameters:
    free_cash_flow (float): The most recent Free Cash Flow (FCF).
    growth_rate (float): The growth rate of FCF for the forecast period.
    discount_rate (float): The discount rate (WACC).
    terminal_growth_rate (float): The terminal growth rate for perpetuity after forecast.
    years (int): Number of forecast years.
    
    Returns:
    float: The intrinsic value of the stock.
    """
    # Calculate the projected free cash flows
    projected_fcfs = [free_cash_flow * (1 + growth_rate) ** year for year in range(1, years + 1)]
    
    # Discount the projected free cash flows to present value
    discounted_fcfs = [fcf / (1 + discount_rate) ** year for year, fcf in enumerate(projected_fcfs, start=1)]
    
    # Calculate the terminal value (TV)
    terminal_value = (projected_fcfs[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    
    # Discount the terminal value to present value
    discounted_terminal_value = terminal_value / (1 + discount_rate) ** years
    
    # Sum the discounted FCFs and the discounted terminal value
    intrinsic_value = sum(discounted_fcfs) + discounted_terminal_value
    
    return intrinsic_value

tab1, tab2, tab3, tab4 = st.tabs(['Technical Analysis' , "Fundamental Analysis", "Discounted Cash Flow Valuation", "Dividend Discount Valuation"])

with tab1:
    indicators = ['Candlestick Chart', 'Heikin Ashi Candles', 'Engulfing Candles' , 'RSI', 'MACD', 'ATR', 'ADX', 'Supertrend', 'Fast Double Supertrend', 'Slow Double Supertrend', 'SMA Ribbons', 'Bollinger Bands', 'Ichimoku Cloud', 'Fractals']
    default_options = ['Candlestick Chart', 'RSI', 'MACD', 'ATR', 'ADX', 'Supertrend']
    selected_indicators = st.multiselect('Select Indicators', indicators, default = default_options)
    create_plot(df, selected_indicators)

with tab2:
    st.header("Fundamental Analysis")
    statement = st.radio("Select Financial Statement",["Quarterly Income Statement", "Quarterly Balance Sheet", "Quarterly CashFlow Statement", "Insider Transactions", "Institutional Holders","Owership Structure"],horizontal=True)
    if statement == "Quarterly Income Statement":
        st.dataframe(symbol.quarterly_incomestmt)
    elif statement == "Quarterly Balance Sheet":
        st.dataframe(symbol.quarterly_balancesheet)
    elif statement == "Quarterly CashFlow Statement":
        st.dataframe(symbol.quarterly_cashflow)
    elif statement == "Insider Transactions":
        st.dataframe(symbol.insider_transactions)
    elif statement == "Institutional Holders":
        st.dataframe(symbol.institutional_holders)
    elif statement =="Owership Structure":
        st.dataframe(symbol.major_holders)
    else:
        st.write("Select Statement")

with tab3:
    if ticker:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        cash_flow = stock.cashflow
    
    # Extract the most recent Free Cash Flow (FCF)
    if not cash_flow.empty:
        recent_fcf = cash_flow.loc['Free Cash Flow'][-len(cash_flow.loc['Free Cash Flow'])]
        st.write(f"Most Recent Free Cash Flow (FCF): ${recent_fcf:,.2f}")
    else:
        st.write("Unable to retrieve cash flow data.")
    
    # User inputs for DCF parameters
    growth_rate = st.number_input("Growth Rate for FCF (%)", value=5.0) / 100
    discount_rate = st.number_input("Discount Rate (WACC) (%)", value=8.0) / 100
    terminal_growth_rate = st.number_input("Terminal Growth Rate (%)", value=2.0) / 100
    forecast_years = st.number_input("Forecast Period (years)", value=5)
    
    # Calculate intrinsic value using DCF model
    if st.button("Calculate Intrinsic Value (DCF)"):
        intrinsic_value = dcf_valuation(recent_fcf, growth_rate, discount_rate, terminal_growth_rate, forecast_years)
        st.write(f"**Intrinsic Value using Discounted Cash Flow Valuation:** ${intrinsic_value:,.2f}")
    else:
        st.write("Please enter a valid stock ticker symbol.")

with tab4:
    if ticker:
        stock = yf.Ticker(ticker)
    
    # Get the dividend history
    dividends = stock.dividends
    
    # Check if dividend data is available
    if not dividends.empty:
        # Calculate the most recent annual dividend
        recent_dividend = dividends[-4:].sum()  # Assuming quarterly dividends
        
        # Display the most recent annual dividend
        st.write(f"Most Recent Annual Dividend: ${recent_dividend:,.2f}")
        
        # User inputs for growth rate and discount rate
        div_growth_rate = st.number_input("Dividend Growth Rate (%)", value=5.0) / 100
        discount_rate = st.number_input("Discount Rate (%)", value=8.0) / 100
        
        # Calculate intrinsic value using the DDM model
        if st.button("Calculate Intrinsic Value (DDM)"):
            try:
                intrinsic_value = ddm_valuation(recent_dividend, div_growth_rate, discount_rate)
                st.write(f"**Intrinsic Value per Share:** ${intrinsic_value:,.2f}")
            except ZeroDivisionError:
                st.write("Error: Discount rate must be greater than the growth rate.")
        else:
            st.write("This stock does not have a dividend history.")    

