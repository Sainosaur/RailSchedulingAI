import requests
from geopy.geocoders import Nominatim


def get_station_location(station_name):
    geolocator = Nominatim(user_agent="ai_scheduler")
    location = geolocator.geocode(station_name)
    if location:
        return location.latitude, location.longitude
    raise Exception(f"Could not get location for station: {station_name}")


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
            return float(data["results"][0]["elevation"])
    raise Exception(f"Could not get elevation for station: {station_name}")


print(get_station_elevation("34-650 Tymbark, Poland"))
