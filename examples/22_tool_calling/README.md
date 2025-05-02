## Tool Calling

To try out:

1. ```pip install -r requirements.txt```
2. Make sure you have set GEMINI_API_KEY in ../keys.env
3. ```python langgraph/lg_weather_agent.py```

### MCP Client-Server for Geocoding
4. In one terminal, start the server: ```python mcp/mcp_weather_server.py```
5. In another terminal, start the client: ```python mcp/mcp_weather_client.py```

