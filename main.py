from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model('tomato_disease_model.h5')

# Class names from your training
class_names = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to training size
    img = image.resize((256, 256))
    
    # Convert to array - DO NOT normalize here as model has Rescaling layer
    img_array = np.array(img, dtype=np.float32)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the frontend HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tomato Disease Detection</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 100%;
            }
            
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2em;
            }
            
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 20px;
                transition: all 0.3s;
                cursor: pointer;
            }
            
            .upload-area:hover {
                background: #f0f4ff;
                border-color: #764ba2;
            }
            
            #fileInput {
                display: none;
            }
            
            .upload-icon {
                font-size: 48px;
                color: #667eea;
                margin-bottom: 15px;
            }
            
            .upload-text {
                color: #666;
                font-size: 16px;
            }
            
            #imagePreview {
                max-width: 100%;
                max-height: 300px;
                margin: 20px auto;
                display: none;
                border-radius: 10px;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 30px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s;
                font-weight: 600;
            }
            
            .btn:hover {
                transform: translateY(-2px);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            #result {
                margin-top: 30px;
                padding: 20px;
                background: #f0f4ff;
                border-radius: 10px;
                display: none;
            }
            
            .result-title {
                font-size: 18px;
                color: #667eea;
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            .disease-name {
                font-size: 24px;
                color: #333;
                margin-bottom: 10px;
                font-weight: 700;
            }
            
            .confidence {
                font-size: 18px;
                color: #666;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍅 Tomato Disease Detection</h1>
            <p class="subtitle">Upload a tomato leaf image to detect diseases</p>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📤</div>
                <div class="upload-text">Click to upload or drag and drop</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <img id="imagePreview" alt="Preview">
            
            <button class="btn" id="predictBtn" disabled>Analyze Image</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px; color: #666;">Analyzing...</p>
            </div>
            
            <div id="result">
                <div class="result-title">Detection Result:</div>
                <div class="disease-name" id="diseaseName"></div>
                <div class="confidence" id="confidence"></div>
            </div>
        </div>
        
        <script>
            const fileInput = document.getElementById('fileInput');
            const imagePreview = document.getElementById('imagePreview');
            const predictBtn = document.getElementById('predictBtn');
            const result = document.getElementById('result');
            const loading = document.getElementById('loading');
            let selectedFile = null;
            
            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                        predictBtn.disabled = false;
                        result.style.display = 'none';
                    }
                    reader.readAsDataURL(file);
                }
            });
            
            predictBtn.addEventListener('click', async function() {
                if (!selectedFile) return;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                predictBtn.disabled = true;
                loading.style.display = 'block';
                result.style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    document.getElementById('diseaseName').textContent = data.disease;
                    document.getElementById('confidence').textContent = 
                        `Confidence: ${data.confidence}%`;
                    
                    result.style.display = 'block';
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                    predictBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict tomato disease from uploaded image"""
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        disease_name = class_names[predicted_class_idx]
        
        # Format disease name (remove prefix, make readable)
        disease_name = disease_name.replace('Tomato_', '').replace('_', ' ')
        
        return {
            "disease": disease_name,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)