import streamlit as st
import pickle
import numpy as np
import requests

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

API_KEY = "8b1443d66604f68a39662687cb1aeae8"

def fetch_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)
        return temp, humidity, rainfall
    return None, None, None

# Streamlit UI
st.title("🌱 Smart Crop Recommendation System")

st.write("Enter soil nutrients and environmental conditions:")

# User input
city = st.text_input("Enter City (optional, auto-fetches weather)")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)

ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
humidity = st.slider("Humidity (%)", 0, 100, 50)
rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# Predict button
if st.button("🌾 Recommend Crop"):
    if city:
        temp, hum, rain = fetch_weather(city)
        if temp is not None:
            temperature, humidity, rainfall = temp, hum, rain
            st.info(f"Weather fetched for {city}: Temp={temp}°C, Humidity={hum}%, Rainfall={rain}mm")

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    probabilities = model.predict_proba(features)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]

    st.subheader("Top 3 Recommended Crops:")
    for i in top_indices:
        st.write(f"- **{model.classes_[i]}** → {round(probabilities[i]*100, 2)}%")

    # Fertilizer recommendations
    st.subheader("💡 Fertilizer Suggestions:")
    if N < 40:
        st.write("- Low Nitrogen: Add **Urea** or **Compost**")
    if P < 40:
        st.write("- Low Phosphorus: Add **DAP** or **Bone Meal**")
    if K < 40:
        st.write("- Low Potassium: Add **MOP** or **Wood Ash**")
    if ph < 5.5:
        st.write("- Soil is acidic: Add **Lime** treatment")
    if ph > 8.0:
        st.write("- Soil is alkaline: Add **Gypsum**")
