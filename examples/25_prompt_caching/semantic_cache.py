import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import time
import re

from anthropic import Anthropic

from dotenv import load_dotenv

if os.path.exists("examples/keys.env"):
    load_dotenv("examples/keys.env")
else:
    raise FileNotFoundError("examples/keys.env not found")


class PromptCache:
    def __init__(self, cache_dir: str = ".prompt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.num_semantic_requests = 10

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a unique cache key for the prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the path to the cache file."""
        return self.cache_dir / f"{cache_key}.json"

    def get_cached_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if it exists."""
        cache_key = self._get_cache_key(prompt)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def _get_semantic_requests(self, prompt: str, show_alternatives: bool = True) -> List[str]:
      """Use LLM to get {self.num_semantic_requests} semantic, alternative requests for the prompt."""

      claude_prompt = f"""
      You are a helpful assistant that generates {self.num_semantic_requests} semantic, alternative requests for a given prompt.
      The requests should be different from the original prompt, but still related to the same topic.
      The requests should be in the same language as the prompt.

      Here is the prompt:
      {prompt}

      Return a JSON array of {self.num_semantic_requests} requests.
      """
      response = self.client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        messages=[{"role": "user", "content": claude_prompt}]
      )
      try:
          response_text = response.content[0].text

          json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
          alternatives = json.loads(json_match.group(1))
      except json.JSONDecodeError:
          print(f"Error: {response.content[0].text}")
          return []

      if show_alternatives:
          alternative_str = "\n".join([f"*{i+1}* {alt}" for i, alt in enumerate(alternatives)])
          print(f"Alternatives:\n{alternative_str}")
      return alternatives

    def cache_response(self, prompt: str, response: Dict[str, Any]):
        """Cache a response for future use."""
        cache_key = self._get_cache_key(prompt)
        cache_path = self._get_cache_path(cache_key)

        with open(cache_path, 'w') as f:
            json.dump(response, f)

    def get_completion(self, prompt: str, use_cache: bool = True) -> str:
        """Get a completion from Claude, using cache if available."""
        if use_cache:
            cached_response = self.get_cached_response(prompt)
            if cached_response:
                print("Using cached response")
                return cached_response["content"]

        print("Making API call to Claude")
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Cache the response
        response_dict = {
            "content": response.content[0].text,
            "model": response.model,
            "usage": response.usage.dict()
        }
        semantic_requests = self._get_semantic_requests(prompt)
        all_requests = [prompt] + semantic_requests
        for request in all_requests:
            self.cache_response(request, response_dict)

        return response.content[0].text

def main():
    # Initialize the cache
    cache = PromptCache()

    # Example prompt
    prompt = "Explain the concept of prompt caching in AI systems. 100 words max."

    # First call - will make API request
    print("\nFirst call:")
    response1_start_time = time.time()
    response1 = cache.get_completion(prompt)
    response1_end_time = time.time()
    print(response1)

    # Second call - will use cache
    print("\nSecond call:")
    response2_start_time = time.time()
    response2 = cache.get_completion(prompt)
    response2_end_time = time.time()
    print(response2)

    # Verify responses are identical
    print("\nResponses are identical:", response1 == response2)

    print(f"Response time: {response1_end_time - response1_start_time:.2f} seconds")
    print(f"Response time: {response2_end_time - response2_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
