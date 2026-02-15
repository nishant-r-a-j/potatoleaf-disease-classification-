# 🌱 Potato Leaf Disease Detection & Smart Advisory System

An AI-powered web application built using **Flask and TensorFlow** that classifies potato leaf diseases into seven major categories:

## 🦠 Disease Categories
- Bacteria  
- Fungi  
- Healthy  
- Nematode  
- Pest  
- Phytopthora (Late Blight)  
- Virus  

---

## 🚀 Key Features

### ✅ Two-Step AI Validation
1. Leaf Detection Model (Rejects non-leaf images)
2. Disease Classification Model (7-class classifier)

### 🌍 Location-Based Weather Integration
- Fetches real-time weather data
- Provides environmental context for disease spread

### 📚 Knowledge-Based Solutions
- Expert advisory for each disease
- Favorable temperature & climate conditions included

### 🌐 Multilingual Accessibility
- Hindi & English support
- Real-time voice read-aloud feature

### 🎯 Confidence-Based Predictions
- Displays prediction probability for reliability

---

## 🛠 Tech Stack
- Python
- Flask
- TensorFlow / Keras
- NumPy
- Open-Meteo API
- HTML, CSS, JavaScript

---

## ▶️ How to Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
