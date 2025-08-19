import requests

# Ref: https://www.youtube.com/watch?v=s8XzpiWfq9I
def get_coordinates(address, api_key):
    """Convert address to latitude/longitude using Geocoding API"""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(url, params=params).json()

    if response["status"] == "OK":
        location = response["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else:
        raise Exception(f"Geocoding error: {response['status']}")

def search_nearby(lat, lng, place_type, api_key, radius=2000):
    """Search for nearby places of a certain type"""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,  # in meters
        "type": place_type,  # e.g. 'school', 'hospital', 'stadium'
        "key": api_key
    }
    response = requests.get(url, params=params).json()

    if response["status"] == "OK":
        return [place["name"] for place in response["results"]]
    else:
        raise Exception(f"Places API error: {response['status']}")

if __name__ == "__main__":
    API_KEY = "AIzaSyA5RgflszSfDw77e9ibONAmrrGwzCPUVUk"  # <-- Replace with your Google Maps API Key
    address = "Beirut, Lebanon"

    # Step 1: Get lat/lng of the address
    lat, lng = get_coordinates(address, API_KEY)

    # Step 2: Search for nearby places
    categories = ["school", "university", "hospital", "stadium", "tourist_attraction"]

    for category in categories:
        results = search_nearby(lat, lng, category, API_KEY)
        print(f"\nNearby {category.capitalize()}s:")
        for r in results:
            print(f" - {r}")