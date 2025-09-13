
# Test script to verify the fixed API server works
# Save as: test_fixed_api.py

import requests
import json

def test_api():
    """Test the fixed agricultural API"""

    base_url = "http://localhost:5000"

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed:")
            print(f"   Status: {health_data['status']}")
            print(f"   Model Type: {health_data.get('model_type', 'Unknown')}")
            print(f"   Available Crops: {health_data.get('crops', 'Unknown')}")
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

    # Test prediction endpoint
    test_data = {
        "N": 35,
        "P": 18, 
        "K": 125,
        "temperature": 28,
        "humidity": 78,
        "ph": 6.2,
        "rainfall": 160,
        "soil_moisture": 70
    }

    try:
        response = requests.post(f"{base_url}/api/predict", 
                               json=test_data,
                               headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("\n✅ Prediction test passed:")
                print(f"   Primary Crop: {result.get('primary_crop', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 'Unknown')}%")
                print(f"   Top 3: {[rec['crop'] for rec in result.get('top_3_recommendations', [])]}")
                return True
            else:
                print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ API request failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Fixed Agricultural API Server")
    print("=" * 40)
    print("Make sure the server is running first!")
    print("Run: python fixed_agricultural_api_server.py")
    print()

    if test_api():
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check server logs for details.")
