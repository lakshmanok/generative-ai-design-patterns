import openai
import time
import os
from dotenv import load_dotenv

if os.path.exists("examples/saved_keys.env"):
    load_dotenv("examples/saved_keys.env")
else:
    raise FileNotFoundError("examples/saved_keys.env not found")
# Setup API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------- WITHOUT CACHE ---------

print("=== WITHOUT CACHE ===")
os.environ.pop("OPENAI_CACHE_DIR", None)  # Ensure cache is off

start = time.time()
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is the capital of Monaco? Provide a detailed answer."}],
)
elapsed_nocache = time.time() - start
print("Response:", response.choices[0].message.content.strip())
print("Time taken (no cache): {:.2f} seconds".format(elapsed_nocache))


# -------- WITH CACHE ---------

print("\n=== WITH CACHE ===")
os.environ["OPENAI_CACHE_DIR"] = "./oai_cache"

# Warm up: first call
_ = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is the capital of Monaco? Provide a detailed answer."}],
)

# Cached call: should be much faster
start = time.time()
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is the capital of Monaco? Provide a detailed answer."}],
)
elapsed_cache = time.time() - start
print("Response:", response.choices[0].message.content.strip())
print("Time taken (with cache): {:.4f} seconds".format(elapsed_cache))
