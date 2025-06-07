import os
import time
import json
from anthropic import Anthropic
from dotenv import load_dotenv

if os.path.exists("examples/keys.env"):
    load_dotenv("examples/keys.env")
else:
    raise FileNotFoundError("examples/keys.env not found")

def main():
    client = Anthropic()

    # Create a long system prompt that exceeds 1024 tokens
    long_system_prompt = """You are an expert AI assistant with deep knowledge across multiple domains.
    Your responses should be comprehensive, accurate, and well-structured. You should always:
    1. Provide detailed explanations
    2. Include relevant examples
    3. Cite sources when possible
    4. Consider multiple perspectives
    5. Highlight key points
    6. Use clear and concise language
    7. Maintain a professional tone
    8. Be helpful and informative
    9. Acknowledge limitations
    10. Suggest further reading

    Additional guidelines for your responses:
    - Always start with a clear introduction
    - Use bullet points for key information
    - Include relevant statistics when available
    - Explain complex concepts in simple terms
    - Provide historical context when relevant
    - Consider cultural implications
    - Address potential misconceptions
    - Include practical applications
    - Suggest related topics
    - End with a clear conclusion

    Your expertise covers:
    - Science and Technology
    - History and Culture
    - Arts and Literature
    - Business and Economics
    - Health and Medicine
    - Education and Learning
    - Environment and Sustainability
    - Politics and Society
    - Philosophy and Ethics
    - Sports and Recreation

    For each response, you should:
    1. Analyze the question thoroughly
    2. Research the topic if needed
    3. Structure your response logically
    4. Include relevant examples
    5. Provide supporting evidence
    6. Consider different viewpoints
    7. Address potential questions
    8. Suggest related topics
    9. Include practical applications
    10. End with a clear conclusion

    Remember to:
    - Be accurate and precise
    - Use clear language
    - Provide context
    - Include examples
    - Cite sources
    - Consider implications
    - Address limitations
    - Suggest next steps
    - Be helpful and informative
    - Maintain professionalism

    Your responses should demonstrate:
    - Deep understanding
    - Clear communication
    - Logical structure
    - Comprehensive coverage
    - Practical relevance
    - Cultural awareness
    - Ethical consideration
    - Scientific accuracy
    - Historical context
    - Future implications

    When providing information, ensure:
    - Accuracy of facts
    - Clarity of explanation
    - Relevance of examples
    - Depth of analysis
    - Breadth of coverage
    - Practical applicability
    - Cultural sensitivity
    - Ethical consideration
    - Scientific validity
    - Historical accuracy

    Your communication style should be:
    - Professional
    - Clear
    - Concise
    - Engaging
    - Informative
    - Helpful
    - Respectful
    - Accurate
    - Comprehensive
    - Well-structured

    Remember to always:
    - Verify information
    - Provide context
    - Include examples
    - Cite sources
    - Consider implications
    - Address limitations
    - Suggest next steps
    - Be helpful
    - Maintain professionalism
    - Ensure accuracy"""

    # Your prompt
    prompt = "Explain the concept of prompt caching in AI systems."

    # Common parameters for all requests
    common_params = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 4096,
        "system": long_system_prompt,
        "temperature": 0.0,  # Set to 0 for deterministic responses
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "stream": False
    }

    # First call - will make API request
    print("\nFirst call:")
    response1_start_time = time.time()
    response1 = client.messages.create(**common_params)
    response1_end_time = time.time()
    print(f"First response length: {len(response1.content[0].text)}")
    print(f"First response hash: {hash(response1.content[0].text)}")

    # Check for cache creation tokens - using direct attribute access
    if hasattr(response1, 'usage'):
        print(f"Cache creation tokens: {getattr(response1.usage, 'cache_creation_input_tokens', 0)}")
        print(f"Cache read tokens: {getattr(response1.usage, 'cache_read_input_tokens', 0)}")
        print(f"Input tokens: {response1.usage.input_tokens}")
        print(f"Output tokens: {response1.usage.output_tokens}")

    # Second call - should use caching
    print("\nSecond call:")
    response2_start_time = time.time()
    response2 = client.messages.create(**common_params)
    response2_end_time = time.time()
    print(f"Second response length: {len(response2.content[0].text)}")
    print(f"Second response hash: {hash(response2.content[0].text)}")

    # Check for cache read tokens
    if hasattr(response2, 'usage'):
        print(f"Cache creation tokens: {getattr(response2.usage, 'cache_creation_input_tokens', 0)}")
        print(f"Cache read tokens: {getattr(response2.usage, 'cache_read_input_tokens', 0)}")
        print(f"Input tokens: {response2.usage.input_tokens}")
        print(f"Output tokens: {response2.usage.output_tokens}")

    # Third call - should also use caching
    print("\nThird call:")
    response3_start_time = time.time()
    response3 = client.messages.create(**common_params)
    response3_end_time = time.time()
    print(f"Third response length: {len(response3.content[0].text)}")
    print(f"Third response hash: {hash(response3.content[0].text)}")

    # Check for cache read tokens
    if hasattr(response3, 'usage'):
        print(f"Cache creation tokens: {getattr(response3.usage, 'cache_creation_input_tokens', 0)}")
        print(f"Cache read tokens: {getattr(response3.usage, 'cache_read_input_tokens', 0)}")
        print(f"Input tokens: {response3.usage.input_tokens}")
        print(f"Output tokens: {response3.usage.output_tokens}")

    # Verify responses are identical
    print("\nResponse comparisons:")
    print(f"First == Second: {response1.content[0].text == response2.content[0].text}")
    print(f"Second == Third: {response2.content[0].text == response3.content[0].text}")
    print(f"First == Third: {response1.content[0].text == response3.content[0].text}")

    print("\nTiming:")
    print(f"First call time: {response1_end_time - response1_start_time:.2f} seconds")
    print(f"Second call time: {response2_end_time - response2_start_time:.2f} seconds")
    print(f"Third call time: {response3_end_time - response3_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
