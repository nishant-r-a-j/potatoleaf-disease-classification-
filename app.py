import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import requests
from datetime import datetime
import os 

# Flask application setup
app = Flask(__name__)

# --- MODEL 1 CONFIGURATION ---
# Path set to the clean H5 file that successfully loads
MODEL_1_PATH = "models/leaf_model_final_clean1.keras"  
leaf_model = None
classes_leaf = ['not-potato-leaf','pleaf',] # 'not-potato-leaf' is the NON-LEAF class

# --- MODEL LOADING LOGIC ---
try:
    print(f"ATTEMPTING TO LOAD LEAF MODEL 1 from: {MODEL_1_PATH}")
    leaf_model = load_model(MODEL_1_PATH)
    print("SUCCESS: Leaf model loaded successfully. Two-step validation is ACTIVE.")
except Exception as e:
    print("CRITICAL WARNING: Leaf model failed to load. The Two-Step validation is DISABLED!")
    print(f"ERROR DETAILS: {e}")
    print("ACTION REQUIRED: Ensure the file at the path above is the one created by the Sequential model.")
    leaf_model = None

# Load 2nd model (Disease Classifier) - Original path remains
try:
    model2 = load_model("models/model2.h5") 
    print("SUCCESS: Disease model (Model 2) loaded successfully.")
except Exception as e:
    print(f"WARNING: Disease model (Model 2) not loaded. Error: {e}")
    model2 = None 

classes_model2 = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus', 'train']

# --- EXPERT KNOWLEDGE BASE (KB) ---
DISEASE_KB = {
    'Bacteria': {
        'common_name': 'Bacterial Wilt',
        'favorable_temp': '25-30°C',
        'favorable_climate': 'High moisture and warm soil (above 25°C).',
        'solution_en': "Apply Streptomycin Sulfate or Copper-based fungicides early. Ensure proper drainage and avoid reusing soil from infected areas for 2-3 years.",
        'solution_hi': "शुरुआत में स्ट्रेप्टोमाइसिन सल्फेट या कॉपर-आधारित फफूंदीनाशक का उपयोग करें। जल निकासी ठीक करें और संक्रमित मिट्टी का 2-3 साल तक पुन: उपयोग न करें।"
    },
    'Fungi': {
        'common_name': 'Fungal Infection (General)',
        'favorable_temp': '18-28°C',
        'favorable_climate': 'High humidity and long periods of leaf wetness (>6 hours).',
        'solution_en': "Use broad-spectrum systemic fungicides (e.g., Azoxystrobin). Improve air circulation by pruning and avoid late-day watering.",
        'solution_hi': "व्यापक-स्पेक्ट्रम सिस्टमिक फफूंदीनाशक (जैसे एज़ोक्सिस्ट्रोबिन) का प्रयोग करें। छँटाई करके हवा का संचार सुधारें और देर शाम को पानी देने से बचें।"
    },
    'Nematode': {
        'common_name': 'Nematode Infection (Root Knot)',
        'favorable_temp': '20-30°C',
        'favorable_climate': 'Sandy soil and warm climate.',
        'solution_en': "Practice crop rotation with non-host plants (e.g., maize). Apply nematicides like Metam sodium. Use bio-control agents like Paecilomyces lilacinus.",
        'solution_hi': "गैर-मेज़बान पौधों (जैसे मक्का) के साथ फसल चक्र का अभ्यास करें। मेटम सोडियम जैसे नेमाटीसाइड्स का प्रयोग करें। जैविक नियंत्रण एजेंटों का उपयोग करें।"
    },
    'Pest': {
        'common_name': 'Common Potato Pest (e.g., Aphids/Mites)',
        'favorable_temp': '20-35°C',
        'favorable_climate': 'Warm and dry conditions; often spread viruses.',
        'solution_en': "For minor infestations, use Neem oil or insecticidal soap. For severe cases, use systemic insecticides like Imidacloprid. Introduce natural predators like ladybugs.",
        'solution_hi': "मामूली संक्रमण के लिए, नीम का तेल या कीटनाशक साबुन का उपयोग करें। गंभीर मामलों में, इमिडाक्लोप्रिड जैसे सिस्टमिक कीटनाशकों का प्रयोग करें। प्राकृतिक शिकारी (जैसे लेडीबग्स) का उपयोग करें।"
    },
    'Healthy': {
        'common_name': 'Healthy Plant',
        'favorable_temp': 'N/A',
        'favorable_climate': 'N/A',
        'solution_en': "Maintain optimal soil moisture and nutrient levels. Monitor weather and pests regularly.",
        'solution_hi': "मिट्टी की इष्टतम नमी और पोषक तत्वों के स्तर को बनाए रखें। मौसम और कीटों की नियमित रूप से निगरानी करें।"
    },
    'Phytopthora': {
        'common_name': 'Late Blight (Phytophthora Infestans)',
        'favorable_temp': '10-20°C',
        'favorable_climate': 'Cool, humid weather (temp below 20°C and 90%+ humidity).',
        'solution_en': "Immediately apply curative fungicides (e.g., Propamocarb, Metalaxyl). Remove and destroy all infected plant material. Ensure plants are properly hilled.",
        'solution_hi': "तुरंत उपचारात्मक फफूंदीनाशक का प्रयोग करें। सभी संक्रमित पौधों को हटाकर नष्ट कर दें। सुनिश्चित करें कि मिट्टी की भराई (hilling) ठीक से की गई हो।"
    },
    'Virus': {
        'common_name': 'Potato Virus Y (PVY)',
        'favorable_temp': '15-25°C',
        'favorable_climate': 'Any climate; spread rapidly by pests (Aphids).',
        'solution_en': "No chemical cure. Remove and destroy infected plants immediately. Control aphid populations aggressively using mineral oils or insecticides. Use certified disease-free seeds.",
        'solution_hi': "कोई रासायनिक इलाज नहीं है। संक्रमित पौधों को तुरंत हटाकर नष्ट कर दें। खनिज तेलों या कीटनाशकों का उपयोग करके एफिड्स की आबादी को सख्ती से नियंत्रित करें। प्रमाणित रोग-मुक्त बीजों का उपयोग करें。"
    },
    'train': { # Assuming 'train' class is mapped to Early Blight
        'common_name': 'Early Blight (Alternaria Solani)',
        'favorable_temp': '24-29°C',
        'favorable_climate': 'Warm, wet weather and prolonged leaf wetness.',
        'solution_en': "Apply protective fungicides (e.g., Mancozeb or Chlorothalonil) before the onset of disease. Practice crop rotation and water in the morning.",
        'solution_hi': "रोग शुरू होने से पहले सुरक्षात्मक फफूंदीनाशक का प्रयोग करें। फसल चक्र का अभ्यास करें और सुबह के समय पानी दें।"
    }
}


def preprocess(img_bytes):
    # Preprocess function: opens, converts, resizes, normalizes, and adds batch dimension
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Weather/temp/season/location info
def get_weather_info(lat=None, lon=None):
    try:
        # If no coordinates provided, use Delhi default
        if lat is None or lon is None:
            lat, lon = 28.61, 77.23
        res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
        res.raise_for_status() # Check for bad status codes
        data = res.json()
        temp = data['current_weather']['temperature']
        
        # Simple season based on month
        month = datetime.now().month
        if month in [12,1,2]: season="Winter"
        elif month in [3,4,5]: season="Spring"
        elif month in [6,7,8]: season="Summer"
        else: season="Autumn"
        
        return {"temp": temp, "season": season, "lat": lat, "lon": lon, "note": "Weather info from open-meteo"}
    except Exception as e:
        print(f"Weather info fetch failed: {e}")
        return {"temp": "N/A", "season": "N/A", "lat": lat, "lon": lon, "note": "Could not fetch weather info"}

@app.route("/")
def home():
    # Renders the HTML template
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['image']
    img_bytes = file.read()
    img_array = preprocess(img_bytes)
    
    # --- STEP 1: LEAF VALIDATION (CRITICAL STEP) ---
    if leaf_model:
        leaf_pred = leaf_model.predict(img_array)
        # Assuming binary classification: output is a probability (0 to 1)
        prediction_index = int(leaf_pred[0][0] > 0.5) 
        leaf_class = classes_leaf[prediction_index]
        
        # Confidence calculation based on the predicted index
        leaf_conf = float(leaf_pred[0][0]) if prediction_index == 1 else float(1 - leaf_pred[0][0])

        # LOGGING for debugging
        print(f"DEBUG: Model 1 Prediction: {leaf_class} (Index: {prediction_index}) with Confidence: {leaf_conf:.2f}")

        # Checking against the actual class name 'not-potato-leaf'
        if leaf_class == 'not-potato-leaf': 
            print("INFO: Image classified as Non-Leaf. Stopping prediction flow.")
            # Returns detailed messages for the frontend to read out
            return jsonify({
                "error": "Invalid Image", 
                "message_en": "The uploaded image is not classified as a potato leaf. Please upload a clear image of a single leaf.",
                "message_hi": "अपलोड की गई छवि आलू की पत्ती के रूप में वर्गीकृत नहीं हुई है। कृपया केवल एक पत्ती की स्पष्ट छवि अपलोड करें।",
                "leaf_check_status": f"Rejected as Non-Leaf (Class: {leaf_class}, Conf: {leaf_conf:.2f})"
            }), 400
    else:
        # If model failed to load, skip this check
        print("CRITICAL LOG: Skipping Leaf Validation (Model 1 failed to load).")
    
    # --- STEP 2: DISEASE CLASSIFICATION (Runs only if Leaf check passes OR Model 1 is disabled) ---
    if model2:
        pred = model2.predict(img_array)
        class2 = classes_model2[np.argmax(pred)]
        conf2 = float(np.max(pred))
        print(f"DEBUG: Model 2 Prediction: {class2} with Confidence: {conf2:.2f}")
    else:
        # SIMULATION MODE (Used when disease model is not loaded)
        class2 = 'Phytopthora' 
        conf2 = 0.95
        print("WARNING: Using simulated disease prediction.")
    
    # --- STEP 3: WEATHER & KB RETRIEVAL ---
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    if lat: lat = float(lat)
    if lon: lon = float(lon)

    weather_info = get_weather_info(lat, lon)
    disease_info = DISEASE_KB.get(class2, {}) 

    result = {
        "model2_class": class2,
        "model2_conf": conf2,
        "weather": weather_info,
        "disease_kb": disease_info
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)