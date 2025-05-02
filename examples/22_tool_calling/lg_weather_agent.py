from dotenv import load_dotenv
import os
import googlemaps
import requests

from typing import Annotated, Literal, TypedDict
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, MessagesState
# from langgraph.checkpoint import MemorySaver
from langchain_core.messages import HumanMessage

# Choose one
PROVIDER = "Gemini"
# PROVIDER = "OpenAI"

# Load key into the environment
load_dotenv("../keys.env")

os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_KEY']
gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_API_KEY"))

# Define the tools that the agent can use
@tool
def latlon_geocoder(location: str) -> (float, float):
    """Converts a place name such as "Kalamazoo, Michigan" to latitude and longitude coordinates"""
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

@tool
def get_weather_from_nws(latitude: float, longitude: float) -> str:
    """Fetches weather data from the National Weather Service API for a specific geographic location."""
    return retrieve_weather_data(latitude, longitude)


tools = [latlon_geocoder, get_weather_from_nws]

if PROVIDER == "Gemini":
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0).bind_tools(tools)
else:
    model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0).bind_tools(tools)


# Define the function that determines the state transition
def assistant_next_node(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# create the workflow
def create_app():
    # Define a new graph: nodes and edges
    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("assistant")
    workflow.add_conditional_edges("assistant", assistant_next_node)
    workflow.add_edge("tools", "assistant")

    # we don't need state, so we don't create a checkpointer
    checkpointer = None  # MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app

def run_query(app, question: str) -> str:
    system_message = """
    Follow the steps in the example below to retrieve the weather information requested.

    Example:
      Question: What's the weather in Kalamazoo, Michigan?
      Step 1:   The user is asking about Kalamazoo, Michigan.
      Step 2:   Use the latlon_geocoder tool to get the latitude and longitude of Kalmazoo, Michigan.
      Step 3:   latitude, longitude is (42.2917, -85.5872)
      Step 4:   Use the get_weather_from_nws tool to get the weather from the National Weather Service at the latitude, longitude
      Step 5:   The detailed forecast for tonight reads 'Showers and thunderstorms before 8pm, then showers and thunderstorms likely. Some of the storms could produce heavy rain. Mostly cloudy. Low around 68, with temperatures rising to around 70 overnight. West southwest wind 5 to 8 mph. Chance of precipitation is 80%. New rainfall amounts between 1 and 2 inches possible.'
      Answer:   It will rain tonight. Temperature is around 70F.

    Question: 
    """
    final_state = app.invoke(
        {"messages": [HumanMessage(content=f"{system_message} {question}")]},
        # config={"configurable": {"thread_id": thread_id}}
    )
    return [m.content.strip() for m in final_state["messages"]]

if __name__ == '__main__':
    app = create_app()
    result = run_query(app, "Is it raining in Chicago?")
    print('\n'.join(result))

