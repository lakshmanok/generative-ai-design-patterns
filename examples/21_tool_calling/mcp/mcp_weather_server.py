from dotenv import load_dotenv
import os
import googlemaps
import requests

from typing import Annotated, Literal, TypedDict
from mcp.server.fastmcp import FastMCP

# Load key into the environment
load_dotenv("../keys.env")

os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_KEY']
gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_API_KEY"))

mcp = FastMCP("weather")

# Define the tools that the agent can use
@mcp.tool()
async def latlon_geocoder(location: str) -> (float, float):
    """Converts a place name such as "Kalamazoo, Michigan" to latitude and longitude coordinates"""
    print(f"Geocoding {location} using Google Maps API")
    geocode_result = gmaps.geocode(location)
    return (round(geocode_result[0]['geometry']['location']['lat'], 4),
            round(geocode_result[0]['geometry']['location']['lng'], 4))


def retrieve_weather_data(latitude: float, longitude: float) -> str:
    """Fetches weather data from the National Weather Service API for a specific geographic location."""
    base_url = "https://api.weather.gov/points/"
    points_url = f"{base_url}{latitude},{longitude}"
    
    headers = {
        "User-Agent": "(weather_agent, vlakshman.com)"
    }  # Replace with your app info

    try:
        print(f"Invoking {points_url}")
        response = requests.get(points_url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
        metadata = response.json()
        # Access specific properties (adjust based on the API response structure)
        forecast_url = metadata.get("properties", {}).get("forecast")
        
        print(f"Invoking {forecast_url}")
        response = requests.get(forecast_url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
        weather_data = response.json()
        return weather_data.get('properties', {}).get("periods")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    return None    

@mcp.tool()
async def get_weather_from_nws(latitude: float, longitude: float) -> str:
    """Fetches weather data from the National Weather Service API for a specific geographic location."""
    return retrieve_weather_data(latitude, longitude)

if __name__ == '__main__':
    mcp.run(transport="sse")
    

