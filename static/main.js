// ----------------------------
// Drag & Drop + File Input
// ----------------------------
const dropArea = document.getElementById("dropArea");
const imgInput = document.getElementById("imgInput");
const preview = document.getElementById("preview");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const resultDiv = document.getElementById("result"); // Result div reference

// Drag & Drop
dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    // Assuming the updated CSS handles the 'highlight' class for better animation
    dropArea.classList.add('highlight'); 
});
dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove('highlight');
});
dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove('highlight');
    const file = e.dataTransfer.files[0];
    showPreview(file);
});

// File input change
imgInput.addEventListener("change", () => {
    if(imgInput.files.length > 0){
        showPreview(imgInput.files[0]);
    }
});

// Show image preview
function showPreview(file){
    const reader = new FileReader();
    reader.onload = function(e){
        preview.src = e.target.result;
        preview.file = file; // save file for prediction
    }
    reader.readAsDataURL(file);
}

// ----------------------------
// Camera Capture
// ----------------------------
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => video.srcObject = stream)
.catch(err => console.error("Camera access denied:", err));

captureBtn.addEventListener("click", () => {
    // Ensure the canvas resolution matches the expected input size (e.g., 224x224)
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
        preview.src = URL.createObjectURL(blob);
        preview.file = blob; // save blob for prediction
    }, 'image/jpeg');
});

// ----------------------------
// Prediction + Weather + Location
// ----------------------------
async function predict() {
    let file = preview.file;
    // Get current language for localized alerts
    const currentLang = document.getElementById('language-select').value;
    
    if(!file) { 
        alert(currentLang === 'en' ? "Please select or capture an image!" : "कृपया एक इमेज चुनें या कैप्चर करें!"); 
        return; 
    }
    
    // Show loader before starting the prediction process
    showLoader();

    const formData = new FormData();
    formData.append("image", file);

    // Get client location
    if(navigator.geolocation){
        navigator.geolocation.getCurrentPosition(
            (position) => {
                formData.append("lat", position.coords.latitude);
                formData.append("lon", position.coords.longitude);
                sendPrediction(formData);
            },
            (err) => {
                console.warn("Geolocation denied or failed, sending without location data.", err);
                // Continue prediction even if geolocation fails
                sendPrediction(formData);
            }
        );
    } else {
        sendPrediction(formData);
    }
}

async function sendPrediction(formData){
    let lat = formData.get("lat");
    let lon = formData.get("lon");

    // Reverse geocode to get city/town
    let locationStr = "Unknown location";
    const currentLang = document.getElementById('language-select').value;
    
    // Translation helpers for results
    const t = (key) => translations[key][currentLang];
    const translateResult = {
        Prediction: t('result_prediction') || "Prediction",
        Temperature: t('result_temp') || "Temperature",
        Season: t('result_season') || "Season",
        Location: t('result_location') || "Location",
        Note: t('result_note') || "Note",
        Unknown: t('result_unknown_loc') || "Unknown location"
    };

    if(lat && lon){
        try {
            const locRes = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}`);
            const locData = await locRes.json();
            locationStr = locData.address.city || locData.address.town || locData.address.village || locData.display_name || translateResult.Unknown;
        } catch(e){
            console.warn("Could not fetch city name", e);
            locationStr = translateResult.Unknown;
        }
    } else {
        locationStr = translateResult.Unknown;
    }

    try {
        // Send image to server
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();

        // Build result HTML
        let html = `<h3>${translateResult.Prediction}: ${data.model2_class} (${(data.model2_conf*100).toFixed(2)}%)</h3>`;
        
        if(data.weather){
            html += `<p>${translateResult.Temperature}: ${data.weather.temp}°C</p>`;
            html += `<p>${translateResult.Season}: ${data.weather.season}</p>`;
            html += `<p>${translateResult.Location}: ${locationStr}</p>`;
            html += `<p>${translateResult.Note}: ${data.weather.note}</p>`;
        }

        // Hide loader and display result
        hideLoader();
        resultDiv.innerHTML = html;

    } catch (error) {
        console.error("Prediction failed:", error);
        // Hide loader and show error message
        hideLoader();
        resultDiv.innerHTML = `<h3 style="color: red;">${currentLang === 'en' ? 'Error: Prediction failed.' : 'त्रुटि: अनुमान विफल रहा।'}</h3><p>${error.message}</p>`;
    }
}