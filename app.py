# app.py
import os
import pickle
import numpy as np
import requests
from flask import Flask, render_template, request

# Load trained model (crop_model.pkl must be in same folder)
MODEL_FILE = "crop_model.pkl"
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}. Run model_training.py first.")

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# OpenWeather API key (set as environment variable OPENWEATHER_API_KEY or paste here)
API_KEY = "8b1443d66604f68a39662687cb1aeae8" 

app = Flask(__name__, template_folder=".")

def recommend_fertilizer(N, P, K, ph):
    recs = []
    # thresholds are conservative example values — tweak if you want
    if N < 40:
        recs.append("Add Urea (high N)")
    if P < 40:
        recs.append("Add DAP/SSP (phosphorus source)")
    if K < 40:
        recs.append("Add MOP (potassium source)")
    if ph < 5.5:
        recs.append("Apply Lime (to raise pH)")
    elif ph > 7.5:
        recs.append("Apply Elemental Sulfur (to lower pH)")
    if not recs:
        recs = ["Soil nutrients appear balanced"]
    return recs

def fetch_weather(city):
    """Return (temperature_C, humidity_pct, rainfall_mm) or (None, None, None) if fail."""
    if not API_KEY:
        return None, None, None
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None, None, None
        data = r.json()
        temp = data.get("main", {}).get("temp")
        humidity = data.get("main", {}).get("humidity")
        # rainfall: try 1h then 3h then 0
        rain = data.get("rain", {}).get("1h", None)
        if rain is None:
            rain = data.get("rain", {}).get("3h", 0)
        if rain is None:
            rain = 0
        return temp, humidity, rain
    except Exception:
        return None, None, None

def safe_float(form, name, default=None):
    v = form.get(name, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default

@app.route("/", methods=["GET"])
def home():
    # initial page load: no values
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    city = form.get("city", "").strip() or None
    soil_preset = form.get("soil_preset", "") or None

    # If a city is provided, try to fetch weather
    temperature, humidity, rainfall = None, None, None
    if city:
        temperature, humidity, rainfall = fetch_weather(city)
        if temperature is None:
            # couldn't fetch — let user know, but allow manual values if present
            error = f"Could not fetch weather for '{city}'. Either API key missing/invalid or city not found. You can still enter values manually."
            # continue and allow manual override
        else:
            error = None
    else:
        error = None

    # Parse values, prefer API values when available and user provided city.
    # For temperature/humidity/rainfall: if API returned None, fallback to form values
    if temperature is None:
        temperature = safe_float(form, "temperature", 25.0)
    if humidity is None:
        humidity = safe_float(form, "humidity", 70.0)
    if rainfall is None:
        rainfall = safe_float(form, "rainfall", 100.0)

    # Nutrients & pH (N,P,K,ph) — these should come from preset autofill or manual inputs
    N = safe_float(form, "N", None)
    P = safe_float(form, "P", None)
    K = safe_float(form, "K", None)
    ph = safe_float(form, "ph", 6.5)

    # if any of N,P,K missing, return with error
    if N is None or P is None or K is None:
        return render_template("index.html", error="Please provide N, P and K values (use a soil preset or enter manually).",
                               city=city, soil_preset=soil_preset,
                               temperature=temperature, humidity=humidity, rainfall=rainfall,
                               N=N, P=P, K=K, ph=ph)

    # Prepare features for model (must match training order)
    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        # top-3 probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            top_idx = np.argsort(probs)[::-1][:3]  # top 3
            top_crops = [(model.classes_[i], round(float(probs[i]) * 100, 2)) for i in top_idx]
        else:
            # fallback: model doesn't have predict_proba
            pred = model.predict(features)[0]
            top_crops = [(pred, 100.0)]
    except Exception as e:
        return render_template("index.html", error=f"Model error: {e}")

    fertilizer_recs = recommend_fertilizer(N, P, K, ph)

    return render_template("index.html",
                           top_crops=top_crops,
                           fertilizer_recs=fertilizer_recs,
                           city=city,
                           soil_preset=soil_preset,
                           N=N, P=P, K=K,
                           temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall,
                           error=error)

if __name__ == "__main__":
    # debug True is convenient locally; switch to False when deploying
    app.run(debug=True)
