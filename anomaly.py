import streamlit as st
import influxdb_client_3 as InfluxDBClient3
import pandas as pd
import numpy as np
from influxdb_client_3 import flight_client_options
import certifi
from adtk.detector import QuantileAD
from adtk.data import validate_series
import matplotlib.pyplot as plt
import telegram
from sklearn.ensemble import IsolationForest
import pygwalker as pyg
import os
import ollama


# Import pandasai and LiteLLM integration
import pandasai as pai
from pandasai_litellm import LiteLLM
from pandasai import SmartDataframe



# Set up API keys as environment variables
# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# Choose LLM model
llm = LiteLLM(model="gpt-3.5-turbo")  # Change to "claude-2" for Anthropic
pai.config.set({"llm": llm})

# Streamlit UI
st.set_page_config(page_title="Real-time Anomaly Detection & Dashboard", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["📊 Anomaly Detection", "📈 Dashboard", "💬 Chat with AI"])

# Telegram configuration
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

def send_telegram_alert(message):
    """Send an alert via Telegram."""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        st.error(f"Failed to send Telegram alert: {e}")

# Read certificate
with open(certifi.where(), "r") as fh:
    cert = fh.read()

# Connect to InfluxDB
client = InfluxDBClient3.InfluxDBClient3(
    token="TBftI4Pi_5RdUWTfoDYxizxPus3WtvW53PHJbX6rtRuUxHipTYrFGDPrfpC2HzLhTOaKsehStaJASvlAc-YFew==",
    org="asksolutions",
    host="https://us-east-1-1.aws.cloud2.influxdata.com",
    database="Temperature Monitoring",
    flight_client_options=flight_client_options(tls_root_certs=cert)
)

def fetch_data():
    """Fetch data from InfluxDB."""
    query = """
        SELECT * FROM "temp"
        WHERE time >= now() - interval '3 day'
        AND ("temp" IS NOT NULL)
    """
    table = client.query(query=query, language="sql")
    df = table.to_pandas()

    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df = df.loc[~df.index.duplicated(keep='first')]
    return df

# Anomaly Detection Page
if page == "📊 Anomaly Detection":
    st.title("📊 Real-time Anomaly Detection with InfluxDB")

    model_choice = st.sidebar.selectbox("Select Anomaly Detection Model", ["QuantileAD", "IsolationForest"])
    enable_alerts = st.sidebar.checkbox("Enable Telegram Alerts", value=False)

    chart_placeholder = st.empty()
    data_placeholder = st.empty()

    df = fetch_data()

    if df is None:
        st.warning("⚠️ No data retrieved from InfluxDB.")
    else:
        if "temp" in df.columns:
            data_temp = df["temp"]
            data_temp = validate_series(data_temp)

            if model_choice == "QuantileAD":
                detector = QuantileAD(low=0.01, high=0.99)
                anomalies = detector.fit_detect(data_temp)
            else:
                X = data_temp.values.reshape(-1, 1)
                iso = IsolationForest(contamination=0.01, random_state=42)
                y_pred = iso.fit_predict(X)
                anomalies = pd.Series(y_pred, index=data_temp.index) == -1

            if enable_alerts and anomalies.any():
                send_telegram_alert("🚨 Anomaly detected in Temperature!")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data_temp.index, data_temp, label="Temperature", color="blue")
            ax.scatter(anomalies[anomalies].index, data_temp[anomalies], color="red", label="Anomalies", marker="x")
            ax.set_title("Real-time Anomaly Detection")
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature")
            ax.legend()
            chart_placeholder.pyplot(fig)

        data_placeholder.dataframe(df)

# Dashboard Page
elif page == "📈 Dashboard":
    st.title("📈 Interactive Data Dashboard")
    df = fetch_data()
    if df is None or df.empty:
        st.warning("⚠️ No data available for visualization.")
    else:
        pyg_html = pyg.walk(df, return_html=True)
        st.components.v1.html(pyg_html, height=800, scrolling=True)

# AI Chat Page
elif page == "💬 Chat with AI":
    st.title("💬 Chat with AI (Powered by Ollama & pandasai)")
    df = fetch_data()
    if df is None or df.empty:
        st.warning("⚠️ No data available for AI analysis.")
    else:
        df_text = df.head(50).to_string(index=False)  # Only send the first 100 rows
        user_input = st.text_area("Ask something about the data:")
        if st.button("Ask AI"):
            if user_input:
                with st.spinner("Thinking..."):
                    # Create a structured prompt with the DataFrame
                    prompt = f"""
                    You are an expert data analyst. Here is the dataset:

                    {df_text}
                    Based on this data, answer the following question:
                    {user_input}
                    """
                    #pandas_ai = pai.PandasAI(llm)
                    #sdf = SmartDataframe(df)
                    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])

                    # Extract only the message content
                    if "message" in response:
                        st.write("🤖 AI Response:")
                        st.write(response["message"]["content"])
                    else:
                        st.write("⚠️ No response message received.")
