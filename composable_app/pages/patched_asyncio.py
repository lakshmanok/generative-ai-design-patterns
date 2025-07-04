import nest_asyncio
import asyncio

# Based on https://sehmi-conscious.medium.com/got-that-asyncio-feeling-f1a7c37cab8b

# Monkey patch Streamlit's internal event loop
nest_asyncio.apply()

def run(x):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(x)
