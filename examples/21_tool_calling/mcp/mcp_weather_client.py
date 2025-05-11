from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio

# Load key into the environment
load_dotenv("../keys.env")


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

async def main():
    async with MultiServerMCPClient(
        {
            "weather": {
                # Ensure your start your weather server on port 8000
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    ) as client:
        agent = create_react_agent(
            "anthropic:claude-3-7-sonnet-latest",
            client.get_tools(),
            prompt = system_message # optional: ReAct works just fine without this
        )
        
        user_input = "Is it raining now in Chicago?"
        while (user_input != "STOP"):
            print(f"Invoking agent for {user_input}")
            weather_response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            print(weather_response['messages'][-1].content)
            print("Type STOP to exit")
            user_input = input("Q: ")
            

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    