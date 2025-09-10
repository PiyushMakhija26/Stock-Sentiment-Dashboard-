import streamlit as st
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from pygooglenews import GoogleNews
import google.generativeai as genai
import textwrap
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Gemini API Configuration ---

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # (Removed model listing for cleaner UI)
    # Try using the most likely model name
    model = genai.GenerativeModel('models/gemini-2.5-flash')
except (KeyError, AttributeError):
    st.error("🚨 Gemini API Key not found. Please add it to your Streamlit secrets.", icon="🚨")
    model = None

def get_sentiment(text):
    """Analyzes the sentiment of a given text."""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def get_stock_list():
    """Returns a list of Nifty 500 stock tickers."""
    # A comprehensive list of Nifty 500 stocks. For brevity, this list is truncated.
    # In a real application, you would load this from a file or API.
    return sorted([
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'BHARTIARTL.NS', 'HINDUNILVR.NS',
        'SBIN.NS', 'LICI.NS', 'ITC.NS', 'HCLTECH.NS', 'LT.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'ADANIENT.NS', 'TITAN.NS', 'ONGC.NS', 'TATAMOTORS.NS', 'AXISBANK.NS', 'NTPC.NS',
        'WIPRO.NS', 'DMART.NS', 'ADANIGREEN.NS', 'M&M.NS', 'ULTRACEMCO.NS', 'BAJAJFINSV.NS', 'ADANIPORTS.NS',
        'POWERGRID.NS', 'TATASTEEL.NS', 'COALINDIA.NS', 'NESTLEIND.NS', 'ASIANPAINT.NS', 'HINDALCO.NS',
        'JSWSTEEL.NS', 'GRASIM.NS', 'INDUSINDBK.NS', 'SBILIFE.NS', 'PIDILITIND.NS', 'TECHM.NS', 'VEDL.NS',
        'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'ADANIENSOL.NS', 'DRREDDY.NS', 'TATACONSUM.NS', 'HDFCLIFE.NS',
        'DIVISLAB.NS', 'BRITANNIA.NS', 'UPL.NS', 'CIPLA.NS', 'HEROMOTOCO.NS', 'SHREECEM.NS', 'APOLLOHOSP.NS',
        'SIEMENS.NS', 'GAIL.NS', 'INDIGO.NS', 'SBICARD.NS', 'LTIM.NS', 'AMBUJACEM.NS', 'ICICIPRULI.NS',
        'HAVELLS.NS', 'IOC.NS', 'BANKBARODA.NS', 'CHOLAFIN.NS', 'DLF.NS', 'BPCL.NS', 'PNB.NS', 'TRENT.NS',
        'SRF.NS', 'GODREJCP.NS', 'TATAPOWER.NS', 'MARICO.NS', 'BERGEPAINT.NS', 'ICICIGI.NS', 'DABUR.NS',
        'BEL.NS', 'HDFCAMC.NS', 'JINDALSTEL.NS', 'ZOMATO.NS', 'MUTHOOTFIN.NS', 'TVSMOTOR.NS', 'COLPAL.NS',
        'NAUKRI.NS', 'ACC.NS', 'UNIONBANK.NS', 'HAL.NS', 'SAIL.NS', 'BHEL.NS', 'IRCTC.NS', 'PAYTM.NS',
        'ZYDUSLIFE.NS', 'VBL.NS', 'MOTHERSON.NS', 'AUROPHARMA.NS', 'UNITDSPR.NS', 'INDUSTOWER.NS',
        'BOSCHLTD.NS', 'HINDPETRO.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'CANBK.NS', 'ABB.NS', 'PETRONET.NS',
        'JSWENERGY.NS', 'MRF.NS', 'HINDZINC.NS', 'MCDOWELL-N.NS', 'TORNTPOWER.NS', 'LUPIN.NS', 'UBL.NS',
        'TATATECH.NS', 'IRFC.NS', 'RVNL.NS', 'AUBANK.NS', 'YESBANK.NS', 'ADANIPOWER.NS', 'POLYCAB.NS'
    ])

def main():
    """Main function to run the Streamlit web app."""
    st.set_page_config(page_title="Advanced Indian Stock Analysis", page_icon="🧠", layout="wide")

    st.title("🧠 Advanced Stock Analysis Dashboard")
    st.markdown("""
    A high-tech dashboard for deep analysis of Indian stocks, combining technicals, fundamentals, news sentiment, and AI insights.
    """)

    # --- Sidebar ---
    with st.sidebar:
        st.header("🔍 Search Stock")
        stock_list = get_stock_list()
        default_index = stock_list.index("RELIANCE.NS") if "RELIANCE.NS" in stock_list else 0
        stock_ticker = st.selectbox("Select Stock Ticker (Nifty 500)", options=stock_list, index=default_index)
        analyze_button = st.button("Analyze Stock")
        st.divider()
        st.header("🤖 FinBot Assistant")
        if model:
            if "chat" not in st.session_state:
                st.session_state.chat = model.start_chat(history=[])
            # Display chat history without duplicating on rerun
            for message in st.session_state.chat.history:
                with st.chat_message("You" if message.role == "user" else "FinBot"):
                    st.markdown(message.parts[0].text)
            if prompt := st.chat_input("Ask about markets, stocks, or finance..."):
                st.chat_message("You").markdown(prompt)
                with st.spinner("FinBot is thinking..."):
                    response = st.session_state.chat.send_message(prompt, stream=False)
                    st.chat_message("FinBot").markdown(response.text)
        else:
            st.warning("Chatbot is disabled as the Gemini API key is not configured.")

    # --- Main Panel ---
    if analyze_button:
        if not stock_ticker:
            st.error("Please select a stock ticker.")
            return

        try:
            with st.spinner(f"Running deep analysis for {stock_ticker}..."):
                # --- Data Fetching ---
                ticker = yf.Ticker(stock_ticker)
                history = ticker.history(period="2y") # Fetch 2 years for better TA
                if history.empty:
                    st.error(f"Could not find data for ticker: {stock_ticker}.")
                    return
                info = ticker.info
                gn = GoogleNews(lang='en', country='IN')
                search = gn.search(f'{info.get("shortName", stock_ticker)} stock', when='7d')
                news_articles = search['entries']

                # --- Advanced Technical Analysis ---
                # Add technical indicators using ta
                history['SMA_50'] = ta.trend.sma_indicator(history['Close'], window=50)
                history['SMA_200'] = ta.trend.sma_indicator(history['Close'], window=200)
                history['RSI_14'] = ta.momentum.rsi(history['Close'], window=14)
                macd = ta.trend.macd(history['Close'])
                macd_signal = ta.trend.macd_signal(history['Close'])
                macd_diff = ta.trend.macd_diff(history['Close'])
                history['MACD_12_26_9'] = macd
                history['MACDs_12_26_9'] = macd_signal
                history['MACDh_12_26_9'] = macd_diff
                history['volume_ma_20'] = history['Volume'].rolling(window=20).mean()
                history['high_50d'] = history['High'].rolling(window=50).max()

                # --- Extract Signals ---
                last_row = history.iloc[-1]
                volume_surge = "✅ Surge" if last_row['Volume'] > 1.8 * last_row['volume_ma_20'] else "Normal"
                breakout_signal = "🔥 Breakout" if last_row['Close'] > history['high_50d'].iloc[-2] else "No"
                
                # --- News and Sentiment ---
                df_news = pd.DataFrame()
                if news_articles:
                    df_news = pd.DataFrame([{"Title": a.title, "Published": a.published, "Source": a.source.title} for a in news_articles[:20]])
                    df_news['Sentiment'] = df_news['Title'].apply(get_sentiment)
                    overall_sentiment = df_news['Sentiment'].value_counts().idxmax() if not df_news.empty else "Neutral"
                else:
                    overall_sentiment = "Neutral"

            st.header(f"{info.get('shortName', stock_ticker)} ({info.get('symbol', '')})")
            
            # --- AI-Powered Summary ---
            if model:
                with st.spinner("🤖 Generating AI-powered analysis summary..."):
                    try:
                        prompt = f"""
                        Analyze the following Indian stock: {info.get('shortName')} ({stock_ticker}).
                        Here is the data:
                        - **Company Profile:** {info.get('longBusinessSummary')}
                        - **Current Price:** ₹{last_row['Close']:.2f}
                        - **Technical Signals:**
                          - RSI (14): {last_row['RSI_14']:.2f}
                          - Trend vs 50-day SMA: {'Above' if last_row['Close'] > last_row['SMA_50'] else 'Below'}
                          - Trend vs 200-day SMA: {'Above' if last_row['Close'] > last_row['SMA_200'] else 'Below'}
                          - Golden/Death Cross: {'Golden Cross (Bullish)' if last_row['SMA_50'] > last_row['SMA_200'] else 'Death Cross (Bearish)'}
                          - 50-Day Breakout Signal: {breakout_signal}
                          - Volume: {volume_surge}
                        - **Fundamental Metrics:**
                          - P/E Ratio: {info.get('trailingPE', 'N/A')}
                          - Debt to Equity: {info.get('debtToEquity', 'N/A')}
                          - Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%
                        - **Recent News Sentiment:** {overall_sentiment}

                        Provide a concise, expert-level summary (3-4 paragraphs) covering:
                        1.  A brief overview of the company.
                        2.  An analysis of the current technical momentum (trend, strength, key signals).
                        3.  A comment on its valuation based on the fundamentals.
                        4.  A conclusion on the overall picture considering the news sentiment.
                        Format the response in Markdown.
                        """
                        response = model.generate_content(prompt)
                        with st.expander("🤖 **View AI-Powered Analysis Summary**", expanded=True):
                            st.markdown(response.text)
                    except Exception as e:
                        st.warning(f"Could not generate AI summary: {e}")

            # --- Tabbed Interface ---
            tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "📑 Fundamental Data", "📰 News & Sentiment"])

            with tab1:
                st.subheader("Interactive Price Chart & Technical Indicators")
                
                # Create figure with secondary y-axis
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                                    row_heights=[0.6, 0.2, 0.2])

                # Candlestick chart
                fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'],
                                             low=history['Low'], close=history['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='purple')), row=1, col=1)

                # RSI chart
                fig.add_trace(go.Scatter(x=history.index, y=history['RSI_14'], mode='lines', name='RSI'), row=2, col=1)
                fig.add_hline(y=70, col=1, row=2, line_dash="dash", line_color="red")
                fig.add_hline(y=30, col=1, row=2, line_dash="dash", line_color="green")

                # MACD chart
                fig.add_trace(go.Scatter(x=history.index, y=history['MACD_12_26_9'], mode='lines', name='MACD'), row=3, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['MACDs_12_26_9'], mode='lines', name='Signal'), row=3, col=1)
                fig.add_trace(go.Bar(x=history.index, y=history['MACDh_12_26_9'], name='Histogram'), row=3, col=1)

                fig.update_layout(height=700, xaxis_rangeslider_visible=False)
                fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)
                fig.update_yaxes(title_text="MACD", row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Key Fundamental Metrics")
                metrics = {
                    "Market Cap": f"₹{info.get('marketCap', 0):,}",
                    "Trailing P/E": info.get('trailingPE'),
                    "Forward P/E": info.get('forwardPE'),
                    "Price to Book": info.get('priceToBook'),
                    "Price to Sales": info.get('priceToSalesTrailing12Months'),
                    "Debt to Equity": info.get('debtToEquity'),
                    "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%",
                    "Beta": info.get('beta'),
                    "52 Week High": f"₹{info.get('fiftyTwoWeekHigh', 0):.2f}",
                    "52 Week Low": f"₹{info.get('fiftyTwoWeekLow', 0):.2f}",
                }
                cols = st.columns(4)
                i = 0
                for label, value in metrics.items():
                    if value is not None:
                        cols[i % 4].metric(label, f"{value:.2f}" if isinstance(value, (int, float)) and label not in ["Market Cap", "Dividend Yield", "52 Week High", "52 Week Low"] else value)
                        i += 1
                
                with st.expander("About Company"):
                    st.write(info.get('longBusinessSummary', 'No summary available.'))

            with tab3:
                st.subheader("Recent News & Sentiment Analysis")
                if not df_news.empty:
                    st.metric(label="Overall News Sentiment", value=overall_sentiment)
                    def color_sentiment(val):
                        return f'color: {"green" if val == "Positive" else "red" if val == "Negative" else "orange"}'
                    st.dataframe(df_news.style.applymap(color_sentiment, subset=['Sentiment']), use_container_width=True)
                else:
                    st.info("No recent news articles found.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()

