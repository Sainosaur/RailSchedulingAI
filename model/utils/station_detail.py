import requests
from geopy.geocoders import Nominatim


def get_station_location(station_name):
    geolocator = Nominatim(user_agent="ai_scheduler")
    location = geolocator.geocode(station_name)
    if location:
        return location.latitude, location.longitude
    return None


def get_station_elevation(station_name):
    location = get_station_location(station_name)
    if location:
        latitude, longitude = location
        url = (
            f"https://api.opentopodata.org/v1/srtm30m?locations={latitude},{longitude}"
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["results"][0]["elevation"]
    return None
