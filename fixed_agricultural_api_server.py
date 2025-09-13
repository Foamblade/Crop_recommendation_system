
# Save this as: fixed_agricultural_api_server.py

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Enhanced HTML template for Jharkhand crops
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Jharkhand Agricultural Decision Support System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f0f8f0; }
        .header { background: linear-gradient(135deg, #4CAF50, #2E7D32); color: white; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; }
        .container { background: white; padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .input-group { margin: 15px 0; }
        .input-row { display: flex; gap: 15px; margin: 15px 0; }
        .input-row .input-group { flex: 1; margin: 0; }
        label { display: block; margin-bottom: 8px; font-weight: bold; color: #2E7D32; }
        input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; }
        input:focus { border-color: #4CAF50; outline: none; box-shadow: 0 0 5px rgba(76, 175, 80, 0.3); }
        button { background: linear-gradient(135deg, #4CAF50, #45a049); color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; font-weight: bold; width: 100%; margin-top: 20px; }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        .result { background: linear-gradient(135deg, #E8F5E8, #C8E6C9); padding: 25px; border-radius: 15px; margin-top: 30px; border-left: 5px solid #4CAF50; }
        .error { background: #ffebee; color: #c62828; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336; }
        .crop-card { background: white; padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #4CAF50; }
        .confidence-bar { background: #ddd; height: 10px; border-radius: 5px; overflow: hidden; margin: 5px 0; }
        .confidence-fill { background: linear-gradient(90deg, #4CAF50, #2E7D32); height: 100%; transition: width 0.3s ease; }
        h1 { margin: 0; font-size: 2.5em; }
        h2 { color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h3 { color: #388e3c; margin-top: 25px; }
        .subtitle { margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em; }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .info-item { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
        .info-value { font-size: 1.5em; font-weight: bold; color: #2E7D32; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #4CAF50; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 Jharkhand Agricultural Decision Support</h1>
        <p class="subtitle">Smart Crop Recommendations for Jharkhand Farmers</p>
        <p class="subtitle">Supporting 12+ crops with AI-powered insights</p>
    </div>

    <div class="container">
        <h2>🚜 Farm Conditions Input</h2>
        <form id="farmForm">
            <div class="input-row">
                <div class="input-group">
                    <label>Nitrogen (N) - kg/ha:</label>
                    <input type="number" id="nitrogen" value="35" step="0.1" min="5" max="80" required>
                </div>
                <div class="input-group">
                    <label>Phosphorus (P) - kg/ha:</label>
                    <input type="number" id="phosphorus" value="18" step="0.1" min="5" max="50" required>
                </div>
                <div class="input-group">
                    <label>Potassium (K) - kg/ha:</label>
                    <input type="number" id="potassium" value="125" step="0.1" min="50" max="200" required>
                </div>
            </div>

            <div class="input-row">
                <div class="input-group">
                    <label>Temperature - °C:</label>
                    <input type="number" id="temperature" value="28" step="0.1" min="10" max="45" required>
                </div>
                <div class="input-group">
                    <label>Humidity - %:</label>
                    <input type="number" id="humidity" value="78" step="0.1" min="30" max="95" required>
                </div>
                <div class="input-group">
                    <label>Soil pH:</label>
                    <input type="number" id="ph" value="6.2" step="0.1" min="4.5" max="9.0" required>
                </div>
            </div>

            <div class="input-row">
                <div class="input-group">
                    <label>Rainfall - mm:</label>
                    <input type="number" id="rainfall" value="160" step="0.1" min="20" max="300" required>
                </div>
                <div class="input-group">
                    <label>Soil Moisture - %:</label>
                    <input type="number" id="moisture" value="70" step="0.1" min="15" max="90" required>
                </div>
            </div>

            <button type="submit">🎯 Get Crop Recommendations</button>
        </form>
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>Analyzing soil and climate conditions...</p>
    </div>

    <div id="results"></div>

    <script>
        document.getElementById('farmForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';

            const data = {
                N: parseFloat(document.getElementById('nitrogen').value),
                P: parseFloat(document.getElementById('phosphorus').value),
                K: parseFloat(document.getElementById('potassium').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                humidity: parseFloat(document.getElementById('humidity').value),
                ph: parseFloat(document.getElementById('ph').value),
                rainfall: parseFloat(document.getElementById('rainfall').value),
                soil_moisture: parseFloat(document.getElementById('moisture').value)
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                // Hide loading
                document.getElementById('loading').style.display = 'none';

                if (result.success) {
                    displayResults(result);
                } else {
                    displayError(result.error || 'Unknown error occurred');
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                displayError('Network error: ' + error.message);
            }
        });

        function displayResults(result) {
            let top3Html = '';
            if (result.top_3_recommendations) {
                result.top_3_recommendations.forEach((rec, index) => {
                    const confidence = rec.confidence || 0;
                    top3Html += `
                        <div class="crop-card">
                            <strong>${index + 1}. ${rec.crop.charAt(0).toUpperCase() + rec.crop.slice(1)}</strong>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidence}%"></div>
                            </div>
                            <small>${confidence.toFixed(1)}% confidence</small>
                        </div>
                    `;
                });
            }

            const html = `
                <div class="result">
                    <h2>🎯 Crop Recommendations for Your Farm</h2>

                    <div style="background: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <h3>🥇 Primary Recommendation</h3>
                        <div style="font-size: 1.8em; color: #2E7D32; font-weight: bold; text-align: center; padding: 15px;">
                            ${result.primary_crop ? result.primary_crop.charAt(0).toUpperCase() + result.primary_crop.slice(1) : 'Not available'}
                        </div>
                        <div style="text-align: center; font-size: 1.2em; color: #666;">
                            ${result.confidence ? result.confidence + '% confidence' : ''}
                        </div>
                    </div>

                    ${top3Html ? '<h3>📊 Top Recommendations</h3>' + top3Html : ''}

                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-value">${result.yield_forecast || 'N/A'}</div>
                            <div>Expected Yield (tons/ha)</div>
                        </div>
                        <div class="info-item">
                            <div class="info-value">₹${result.profit_margin || 'N/A'}</div>
                            <div>Profit Margin</div>
                        </div>
                        <div class="info-item">
                            <div class="info-value">${result.sustainability_score || 'N/A'}/100</div>
                            <div>Sustainability Score</div>
                        </div>
                        <div class="info-item">
                            <div class="info-value" style="color: ${result.risk_level === 'Low' ? '#4CAF50' : result.risk_level === 'Medium' ? '#FF9800' : '#f44336'}">
                                ${result.risk_level || 'N/A'}
                            </div>
                            <div>Risk Level</div>
                        </div>
                    </div>

                    ${result.recommendations && result.recommendations.length > 0 ? `
                        <h3>💡 Farming Recommendations</h3>
                        <ul style="list-style-type: none; padding: 0;">
                            ${result.recommendations.map(rec => `<li style="background: #f8f9fa; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 3px solid #4CAF50;">✓ ${rec}</li>`).join('')}
                        </ul>
                    ` : ''}

                    <div style="margin-top: 30px; padding: 15px; background: #e3f2fd; border-radius: 8px; font-size: 0.9em; color: #1565c0;">
                        <strong>Note:</strong> These recommendations are based on current soil and climate conditions for Jharkhand region. 
                        Consider local market conditions and consult with agricultural extension officers for best results.
                    </div>
                </div>
            `;
            document.getElementById('results').innerHTML = html;
        }

        function displayError(errorMessage) {
            document.getElementById('results').innerHTML = `
                <div class="error">
                    <h3>❌ Error</h3>
                    <p>${errorMessage}</p>
                    <p>Please check your input values and try again.</p>
                </div>
            `;
        }

        // Add some sample data buttons
        function loadSampleData(type) {
            const samples = {
                rice: { N: 35, P: 18, K: 125, temperature: 28, humidity: 78, ph: 6.2, rainfall: 160, moisture: 70 },
                wheat: { N: 25, P: 20, K: 95, temperature: 20, humidity: 55, ph: 6.8, rainfall: 65, moisture: 45 },
                maize: { N: 40, P: 22, K: 110, temperature: 26, humidity: 65, ph: 6.5, rainfall: 100, moisture: 55 }
            };

            const sample = samples[type];
            if (sample) {
                Object.keys(sample).forEach(key => {
                    const element = document.getElementById(key === 'moisture' ? 'moisture' : key);
                    if (element) element.value = sample[key];
                });
            }
        }
    </script>
</body>
</html>
"""

def load_jharkhand_model():
    """Load the enhanced Jharkhand model with proper error handling"""
    try:
        # Try to load the enhanced Jharkhand model first
        with open('enhanced_jharkhand_model.pkl', 'rb') as f:
            models = pickle.load(f)
        print("✅ Loaded enhanced Jharkhand model")
        return models, "enhanced"
    except FileNotFoundError:
        print("⚠️ Enhanced model not found, trying basic model...")
        try:
            with open('jharkhand_agricultural_model.pkl', 'rb') as f:
                models = pickle.load(f)
            print("✅ Loaded basic Jharkhand model")
            return models, "basic"
        except FileNotFoundError:
            print("⚠️ No existing models found, creating synthetic model...")
            return create_synthetic_model(), "synthetic"
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return create_synthetic_model(), "synthetic"

def create_synthetic_model():
    """Create a synthetic model compatible with both old and new structures"""
    print("🔧 Creating enhanced synthetic Jharkhand model...")

    np.random.seed(42)
    n_samples = 1000

    # Generate training data
    X = pd.DataFrame({
        'N': np.clip(np.random.normal(30, 12), 8, 65),
        'P': np.clip(np.random.normal(16, 7), 5, 35),
        'K': np.clip(np.random.normal(115, 25), 70, 180),
        'temperature': np.clip(np.random.normal(26, 7), 12, 42),
        'humidity': np.clip(np.random.normal(68, 16), 35, 95),
        'ph': np.clip(np.random.normal(6.1, 0.9), 4.8, 8.2),
        'rainfall': np.clip(np.random.normal(105, 45), 25, 240),
        'soil_moisture': np.clip(np.random.normal(48, 15), 18, 85)
    })

    # Jharkhand crops with realistic assignment
    crops = []
    for _, row in X.iterrows():
        if (row['ph'] >= 5.5 and row['rainfall'] >= 100 and 
            row['temperature'] >= 22 and row['N'] >= 25):
            crops.append('rice')
        elif (row['ph'] >= 6.0 and row['rainfall'] <= 100 and 
              row['temperature'] <= 25):
            crops.append('wheat')
        elif (row['temperature'] >= 25 and row['P'] >= 15):
            crops.append('maize')
        elif (row['K'] >= 100 and row['temperature'] >= 20):
            crops.append('arhar')
        elif (row['temperature'] <= 25 and row['N'] >= 15):
            crops.append('gram')
        elif (row['temperature'] >= 25 and row['ph'] >= 6.5):
            crops.append('moong')
        elif (row['rainfall'] <= 80 and row['temperature'] <= 25):
            crops.append('mustard')
        elif (row['N'] >= 35 and row['rainfall'] >= 150):
            crops.append('sugarcane')
        elif (row['rainfall'] <= 90):
            crops.append('ragi')
        else:
            crops.append(np.random.choice(['bajra', 'sunflower', 'potato']))

    # Encode labels
    le = LabelEncoder()
    y_crop = le.fit_transform(crops)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    crop_model.fit(X_scaled, y_crop)

    yield_model = RandomForestRegressor(n_estimators=50, random_state=42)
    yields = np.random.normal(4.5, 1.8, len(X))
    yield_model.fit(X_scaled, yields)

    profit_model = RandomForestRegressor(n_estimators=50, random_state=42)
    profits = np.random.normal(15000, 5000, len(X))
    profit_model.fit(X_scaled, profits)

    sust_model = RandomForestRegressor(n_estimators=50, random_state=42)
    sustainability = np.random.normal(78, 12, len(X))
    sust_model.fit(X_scaled, sustainability)

    # Return model compatible with both structures
    return {
        'model': crop_model,          # For new structure
        'crop_model': crop_model,     # For old structure
        'yield_model': yield_model,
        'profit_model': profit_model,
        'sustainability_model': sust_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_columns': list(X.columns)
    }

# Load the appropriate model
models, model_type = load_jharkhand_model()

class EnhancedAgriculturalAPI:
    def __init__(self):
        # Handle both new and old model structures
        if 'model' in models:
            # New enhanced structure
            self.crop_model = models['model']
            self.scaler = models['scaler']
            self.label_encoder = models['label_encoder']
            self.feature_columns = models['feature_columns']
        else:
            # Old structure fallback
            self.crop_model = models.get('crop_model')
            self.scaler = models.get('scaler')
            self.label_encoder = models.get('label_encoder')
            self.feature_columns = models.get('feature_columns')

        # Additional models for comprehensive predictions
        self.yield_model = models.get('yield_model')
        self.profit_model = models.get('profit_model')
        self.sustainability_model = models.get('sustainability_model')

        print(f"🤖 API initialized with {model_type} model")
        print(f"📊 Features: {self.feature_columns}")
        print(f"🌾 Available crops: {len(self.label_encoder.classes_) if self.label_encoder else 'Unknown'}")

    def predict_comprehensive(self, data):
        try:
            # Validate input data
            for feature in self.feature_columns:
                if feature not in data:
                    return {'success': False, 'error': f'Missing required field: {feature}'}

            # Prepare input
            input_array = np.array([[data[feature] for feature in self.feature_columns]])
            input_scaled = self.scaler.transform(input_array)

            # Make crop prediction
            crop_pred = self.crop_model.predict(input_scaled)[0]
            crop_proba = self.crop_model.predict_proba(input_scaled)[0]

            primary_crop = self.label_encoder.classes_[crop_pred]
            confidence = max(crop_proba) * 100

            # Get top 3 recommendations
            top_3_indices = np.argsort(crop_proba)[-3:][::-1]
            top_3_recommendations = []

            for idx in top_3_indices:
                crop_name = self.label_encoder.classes_[idx]
                conf = crop_proba[idx] * 100
                top_3_recommendations.append({
                    'crop': crop_name,
                    'confidence': round(conf, 1)
                })

            # Additional predictions (if models available)
            yield_pred = None
            profit_pred = None
            sustainability_pred = None

            if self.yield_model:
                yield_pred = self.yield_model.predict(input_scaled)[0]
            if self.profit_model:
                profit_pred = self.profit_model.predict(input_scaled)[0]
            if self.sustainability_model:
                sustainability_pred = self.sustainability_model.predict(input_scaled)[0]

            # Risk assessment
            risk_factors = 0
            if data['rainfall'] < 50 or data['rainfall'] > 200:
                risk_factors += 1
            if data['temperature'] < 15 or data['temperature'] > 35:
                risk_factors += 1
            if data['ph'] < 5.5 or data['ph'] > 8.0:
                risk_factors += 1
            if data.get('N', 30) < 20:
                risk_factors += 1

            risk_level = 'Low' if risk_factors <= 1 else 'Medium' if risk_factors == 2 else 'High'

            # Generate recommendations
            recommendations = []
            if data.get('N', 30) < 25:
                recommendations.append('Apply nitrogen fertilizer (40-60 kg/ha) for better crop growth')
            if data.get('P', 15) < 15:
                recommendations.append('Apply phosphorus fertilizer (20-30 kg/ha) for root development')
            if data.get('K', 100) < 90:
                recommendations.append('Apply potassium fertilizer (30-50 kg/ha) for disease resistance')
            if data.get('soil_moisture', 50) < 40:
                recommendations.append('Increase irrigation frequency or install drip irrigation')
            if data.get('ph', 6.5) < 6.0:
                recommendations.append('Apply lime (200-500 kg/ha) to reduce soil acidity')
            if data.get('ph', 6.5) > 7.5:
                recommendations.append('Apply organic matter to reduce soil alkalinity')

            # Crop-specific recommendations
            if primary_crop == 'rice':
                recommendations.append('Maintain water levels 2-5 cm during vegetative stage')
            elif primary_crop == 'wheat':
                recommendations.append('Ensure adequate drainage for optimal wheat growth')
            elif primary_crop in ['arhar', 'gram', 'moong']:
                recommendations.append('Inoculate seeds with Rhizobium for better nitrogen fixation')

            if not recommendations:
                recommendations.append('Current conditions are suitable for the recommended crop')

            return {
                'success': True,
                'primary_crop': primary_crop,
                'confidence': round(confidence, 1),
                'top_3_recommendations': top_3_recommendations,
                'yield_forecast': round(yield_pred, 2) if yield_pred else 4.5,
                'profit_margin': round(profit_pred, 0) if profit_pred else 12000,
                'sustainability_score': round(sustainability_pred, 1) if sustainability_pred else 75,
                'risk_level': risk_level,
                'recommendations': recommendations,
                'model_type': model_type
            }

        except Exception as e:
            return {'success': False, 'error': f'Prediction error: {str(e)}'}

# Initialize API
agri_api = EnhancedAgriculturalAPI()

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_moisture']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                'success': False, 
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Make prediction
        result = agri_api.predict_comprehensive(data)
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': f'API error: {str(e)}'}), 500

@app.route('/api/crops')
def get_crops():
    """Get list of available crops"""
    try:
        crops = agri_api.label_encoder.classes_.tolist()
        return jsonify({
            'success': True,
            'crops': crops,
            'total': len(crops)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'message': 'Enhanced Jharkhand Agricultural API is running',
        'model_type': model_type,
        'crops': len(agri_api.label_encoder.classes_) if agri_api.label_encoder else 0
    })

@app.route('/api/sample-data')
def get_sample_data():
    """Get sample input data for testing"""
    samples = {
        'rice_ranchi': {
            'N': 35, 'P': 18, 'K': 125, 'temperature': 28,
            'humidity': 78, 'ph': 6.2, 'rainfall': 160, 'soil_moisture': 70,
            'description': 'Ideal conditions for rice in Ranchi area'
        },
        'wheat_palamu': {
            'N': 25, 'P': 20, 'K': 95, 'temperature': 20,
            'humidity': 55, 'ph': 6.8, 'rainfall': 65, 'soil_moisture': 45,
            'description': 'Wheat growing conditions in Palamu district'
        },
        'pulse_hazaribagh': {
            'N': 18, 'P': 15, 'K': 85, 'temperature': 24,
            'humidity': 60, 'ph': 6.5, 'rainfall': 80, 'soil_moisture': 50,
            'description': 'Pulse crops suitable for Hazaribagh region'
        }
    }
    return jsonify({'success': True, 'samples': samples})

if __name__ == '__main__':
    print("🌾 Enhanced Jharkhand Agricultural Decision Support System")
    print("=" * 60)
    print("🎯 Starting Flask API server...")
    print(f"🤖 Model Type: {model_type.upper()}")
    print(f"🌾 Available Crops: {len(agri_api.label_encoder.classes_) if agri_api.label_encoder else 'Unknown'}")
    print("📍 Open your web browser and go to: http://localhost:5000")
    print("🔧 API endpoints:")
    print("   • POST /api/predict - Make crop predictions")
    print("   • GET /api/crops - List available crops")
    print("   • GET /api/sample-data - Get sample input data")
    print("   • GET /health - Health check")
    print("❌ Press Ctrl+C to stop the server")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)
