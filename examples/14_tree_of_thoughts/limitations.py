import os
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv(dotenv_path="examples/keys.env", verbose=True, override=True)

def zero_shot_prompt_generation():
    """
    Generate a prompt for a contract clause analysis.

    Returns:
    --------
    str
        The generated prompt
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = """
I need to optimize our supply chain:

Current situation:
- 3 potential manufacturing locations (Mexico, Vietnam, Poland)
- 4 distribution centers (Atlanta, Chicago, Dallas, Seattle)
- 2 primary shipping methods (air, sea)
- Historical demand fluctuations of ±20%
- Recent disruptions in Asian shipping routes

Follow this thought process:

1. Generate 3 different supply chain configurations
2. For each configuration, explore performance under 3 scenarios:
  a. Normal operations
  b. Major shipping disruption
  c. 30% demand increase
3. Evaluate each path for:
  - Total cost
  - Delivery time reliability
  - Disruption vulnerability
4. Compare the risk-adjusted performance of each path
5. Identify which configuration offers the best balance of cost, speed, and resilience

Think step by step.
    """

    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text

def main():
    # Example usage
    prompt = zero_shot_prompt_generation()

    print(f"Prompt: {prompt}")

if __name__ == "__main__":
    main()
