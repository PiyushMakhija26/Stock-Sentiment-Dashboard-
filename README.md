# 🧠 Chart Pattern Intelligence Dashboard

An AI-powered technical analysis suite for the Indian Stock Market (NSE). This dashboard detects real-time chart patterns, calculates historical success rates, and provides plain-English explanations for traders.

## 🚀 Features

- **Real-time Pattern Detection**: Automatically identifies Bullish/Bearish Breakouts, Double Bottoms, and RSI Divergences.
- **Support & Resistance Hub**: Uses pivot-point clustering to identify key horizontal levels where price action is likely to react.
- **Nifty 500 Scanner**: Scan the top 100 NSE stocks instantly for active breakout opportunities.
- **Plain-English Explanations**: Powered by Google Gemini AI to translate complex technical patterns into actionable insights.
- **Historical Backtesting**: Stock-specific success rates based on a 2-year lookback with custom Reward-to-Risk ratios (>1.5 for short-term).
- **Sentiment Analysis**: Real-time news aggregation and sentiment scoring for an all-around market view.
- **Intelligence Reports**: Export detailed stock analysis as a Markdown report for offline review.

## 🛠 Tech Stack

- **Streamlit**: For the interactive dashboard UI.
- **YFinance**: For real-time and historical NSE market data.
- **SciPy & NumPy**: For signal processing and pattern detection logic.
- **Google Gemini API**: For expert-level analysis and explanations.
- **Pandas TA & TextBlob**: For technical indicators and news sentiment.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PiyushMakhija26/Stock-Sentiment-Dashboard-.git
   cd Stock-Sentiment-Dashboard-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google Gemini API Key:
   - Create a `.streamlit/secrets.toml` file or set the environment variable.
   ```toml
   GOOGLE_API_KEY = "your_api_key_here"
   ```

4. Run the app:
   ```bash
   streamlit run "Stock Sentiment Dashboard .py"
   ```

## 📈 Success Metrics

- **Short-term Success**: Reward-to-Risk ratio > 1.5.
- **Long-term Success**: Reward-to-Risk ratio > 2.0.
- Stop-losses and targets are calculated dynamically based on price volatility.

---
*Disclaimer: This tool is for educational purposes only. Market investments are subject to risk. Always consult with a certified financial advisor.*
