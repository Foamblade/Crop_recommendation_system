
import requests
import json
import time

class SoilGridsDataFetcher:
    """
    Complete working implementation to fetch nitrogen and pH from SoilGrids API
    """

    def __init__(self):
        self.base_url = "https://rest.isric.org/soilgrids/v2.0"
        self.rate_limit_delay = 12  # 5 calls per minute = 12 seconds between calls
        self.last_call_time = 0

    def _respect_rate_limit(self):
        """Ensure we don't exceed the 5 calls per minute rate limit"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            print(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

        self.last_call_time = time.time()

    def fetch_single_property(self, longitude, latitude, property_name, depth="0-5cm"):
        """Fetch a single soil property from SoilGrids"""

        self._respect_rate_limit()

        url = f"{self.base_url}/properties/query"

        params = {
            'lon': longitude,
            'lat': latitude,
            'property': property_name,
            'depth': depth,
            'value': 'mean'
        }

        try:
            print(f"Fetching {property_name} for ({latitude}, {longitude}) at {depth}")
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Navigate through the JSON structure
                if 'properties' in data and 'layers' in data['properties']:
                    layers = data['properties']['layers']

                    for layer in layers:
                        if layer.get('name') == property_name:
                            depths = layer.get('depths', [])

                            for depth_info in depths:
                                if depth_info.get('label') == depth:
                                    values = depth_info.get('values', {})
                                    mean_value = values.get('mean')

                                    if mean_value is not None:
                                        return mean_value

                print(f"No data found for {property_name} at {depth}")
                return None

            else:
                print(f"API Error {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
            print(f"Timeout fetching {property_name}")
            return None
        except Exception as e:
            print(f"Error fetching {property_name}: {e}")
            return None

    def fetch_nitrogen_and_ph(self, longitude, latitude, depth="0-5cm"):
        """
        Fetch both nitrogen and pH data for given coordinates

        Parameters:
        - longitude: Longitude in decimal degrees
        - latitude: Latitude in decimal degrees  
        - depth: Soil depth (default: "0-5cm")

        Returns:
        - Dictionary with nitrogen and pH values
        """

        print(f"\nFetching soil data for coordinates: {latitude}, {longitude}")
        print(f"Depth: {depth}")
        print("-" * 50)

        result = {
            'coordinates': {
                'latitude': latitude,
                'longitude': longitude,
                'depth': depth
            },
            'nitrogen': {
                'value_kg_ha': None,
                'raw_value_cg_kg': None,
                'source': 'SoilGrids'
            },
            'ph': {
                'value': None,
                'raw_value_ph10': None,
                'source': 'SoilGrids'
            },
            'success': False,
            'data_quality': 'No data'
        }

        # Fetch nitrogen data
        nitrogen_raw = self.fetch_single_property(longitude, latitude, 'nitrogen', depth)
        if nitrogen_raw is not None:
            result['nitrogen']['raw_value_cg_kg'] = nitrogen_raw
            result['nitrogen']['value_kg_ha'] = nitrogen_raw * 0.1  # Convert cg/kg to kg/ha
            print(f" Nitrogen: {nitrogen_raw} cg/kg  {result['nitrogen']['value_kg_ha']:.2f} kg/ha")
        else:
            print("Nitrogen: No data available")

        # Fetch pH data
        ph_raw = self.fetch_single_property(longitude, latitude, 'phh2o', depth)
        if ph_raw is not None:
            result['ph']['raw_value_ph10'] = ph_raw
            result['ph']['value'] = ph_raw / 10  # Convert pH*10 to pH
            print(f" pH: {ph_raw}/10  {result['ph']['value']:.2f}")
        else:
            print(" pH: No data available")

        # Determine success and data quality
        nitrogen_available = result['nitrogen']['value_kg_ha'] is not None
        ph_available = result['ph']['value'] is not None

        if nitrogen_available and ph_available:
            result['success'] = True
            result['data_quality'] = 'High (Both N and pH from SoilGrids)'
        elif nitrogen_available or ph_available:
            result['success'] = True
            result['data_quality'] = 'Medium (Partial data from SoilGrids)'
        else:
            result['data_quality'] = 'Low (No SoilGrids data available)'

        print(f"\nData Quality: {result['data_quality']}")
        print("-" * 50)

        return result

    def get_agricultural_input_data(self, longitude, latitude, weather_data=None):
        """
        Get soil data formatted for agricultural ML models
        """

        soil_data = self.fetch_nitrogen_and_ph(longitude, latitude)

        # Format for agricultural system
        input_data = {
            'N': soil_data['nitrogen']['value_kg_ha'] or 40,  # Default if no data
            'ph': soil_data['ph']['value'] or 6.5,           # Default if no data
            'P': 25,  # SoilGrids doesn't provide P, need other source
            'K': 35   # SoilGrids doesn't provide K, need other source
        }

        # Add weather data if provided
        if weather_data:
            input_data.update(weather_data)
        else:
            # Default weather values
            input_data.update({
                'temperature': 25,
                'humidity': 65, 
                'rainfall': 100,
                'soil_moisture': 50
            })

        return {
            'input_data': input_data,
            'data_quality': soil_data['data_quality'],
            'sources': {
                'nitrogen': 'SoilGrids' if soil_data['nitrogen']['value_kg_ha'] else 'Default',
                'ph': 'SoilGrids' if soil_data['ph']['value'] else 'Default',
                'phosphorus': 'Default (SoilGrids unavailable)',
                'potassium': 'Default (SoilGrids unavailable)'
            }
        }

# Test the implementation
if __name__ == "__main__":

    print(" SOILGRIDS NITROGEN & pH DATA FETCHER")
    print("=" * 60)

    # Initialize the fetcher
    fetcher = SoilGridsDataFetcher()

    # Test locations
    test_locations = [
        {"name": "Random, India", "lat": 22.8127, "lon": 85.3656},
        {"name": "Netherlands", "lat": 52.0, "lon": 5.0},
        {"name": "Iowa, USA", "lat": 42.0, "lon": -93.5}
    ]

    for i, location in enumerate(test_locations):
        print(f"\nTEST {i+1}: {location['name']}")

        # Get agricultural input data
        result = fetcher.get_agricultural_input_data(
            longitude=location['lon'], 
            latitude=location['lat'],
            weather_data={'temperature': 25, 'humidity': 60, 'rainfall': 120, 'soil_moisture': 45}
        )

        print("\nAGRICULTURAL INPUT DATA:")
        for param, value in result['input_data'].items():
            source = result['sources'].get(param, 'Default')
            print(f"  {param}: {value} ({source})")

        print(f"\nOverall Data Quality: {result['data_quality']}")

        if i < len(test_locations) - 1:
            print("\n" + "="*60)

    print("\nTESTING COMPLETE!")
    print("\nINTEGRATION TIPS:")
    print("• Use coordinates (lat, lon) to get real soil data")
    print("• System falls back to defaults when SoilGrids data unavailable") 
    print("• Rate limiting automatically handled (5 calls/minute)")
    print("• Data quality indicator helps users understand reliability")
