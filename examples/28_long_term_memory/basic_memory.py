"""
Basic Mem0 Example: LLM Inference with Memory

Prerequisites:
- pip install mem0ai openai python-dotenv chromadb
- OpenAI API key set in environment variable OPENAI_API_KEY

This example demonstrates the core functionality of mem0:
1. Adding information to memory
2. Searching memories for relevant context
3. Using memories to enhance LLM responses

No agent setup - just basic LLM inference enhanced with memory.
"""

import os
import tempfile
from typing import List, Dict, Any
from mem0 import Memory
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if os.path.exists("examples/keys.env"):
    load_dotenv("examples/keys.env")
else:
    raise FileNotFoundError("examples/keys.env not found")

# Set up your API keys
# You can set these as environment variables or replace with your actual keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_memory_config():
    """Get a working memory configuration that avoids permission issues"""
    # Create a temporary directory for the databases
    temp_dir = tempfile.mkdtemp(prefix="mem0_example_")
    print(f"Using temporary directory for mem0 storage: {temp_dir}")

    # Configuration that uses local file storage with proper permissions
    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mem0_basic_example",
                "path": os.path.join(temp_dir, "chroma_db"),
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1500,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        },
        "history_db_path": os.path.join(temp_dir, "history.db"),
        "version": "v1.1"
    }

    return config

def initialize_clients():
    """Initialize mem0 Memory and OpenAI clients with proper configuration"""
    try:
        # Try with custom configuration first
        config = get_memory_config()
        memory = Memory.from_config(config)
        print("✓ Mem0 initialized with custom configuration (Chroma DB)")
    except Exception as e:
        print(f"Warning: Custom config failed ({e}), trying default configuration...")
        try:
            # Fallback to default configuration
            memory = Memory()
            print("✓ Mem0 initialized with default configuration")
        except Exception as e2:
            print(f"Error: Both configurations failed. {e2}")
            print("\nTroubleshooting:")
            print("1. Make sure you have write permissions in the current directory")
            print("2. Install required packages: pip install mem0ai openai chromadb")
            print("3. For production use, consider setting up a proper vector database")
            raise

    # Initialize OpenAI client for LLM inference
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return memory, openai_client

def add_conversation_to_memory(memory: Memory, messages: List[Dict[str, str]], user_id: str) -> Dict[str, Any]:
    """Add a conversation to mem0 memory"""
    print(f"Adding conversation to memory for user: {user_id}")
    try:
        result = memory.add(messages, user_id=user_id)
        print(f"✓ Memory add successful: {result}")
        return result
    except Exception as e:
        print(f"Error adding to memory: {e}")
        return {"error": str(e)}

def search_relevant_memories(memory: Memory, query: str, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Search for relevant memories based on query"""
    print(f"Searching memories for: '{query}'")
    try:
        memories = memory.search(query=query, user_id=user_id, limit=limit)
        print(f"Raw search result type: {type(memories)}")
        print(f"Raw search result: {memories}")

        # Handle different return formats from mem0
        processed_memories = []

        if isinstance(memories, list):
            for i, mem in enumerate(memories):
                print(f"Memory item {i} type: {type(mem)}")

                if isinstance(mem, dict):
                    # Standard dictionary format
                    if 'memory' in mem:
                        processed_memories.append(mem)
                        print(f"  {i+1}. {mem.get('memory', 'No content')}")
                    elif 'text' in mem:
                        # Alternative format with 'text' key
                        processed_memories.append({'memory': mem['text'], **mem})
                        print(f"  {i+1}. {mem.get('text', 'No content')}")
                    else:
                        # Unknown dict format, try to find content
                        content = str(mem)
                        processed_memories.append({'memory': content})
                        print(f"  {i+1}. {content}")

                elif isinstance(mem, str):
                    # String format
                    processed_memories.append({'memory': mem})
                    print(f"  {i+1}. {mem}")
                else:
                    # Other formats, convert to string
                    content = str(mem)
                    processed_memories.append({'memory': content})
                    print(f"  {i+1}. {content}")

        elif isinstance(memories, dict):
            # Check if it's a response with results
            if 'results' in memories:
                return search_relevant_memories_from_results(memories['results'], query)
            else:
                # Single memory dict
                processed_memories = [memories]
                print(f"  1. {memories.get('memory', str(memories))}")
        else:
            print(f"Unexpected memories format: {type(memories)}")
            return []

        print(f"Found {len(processed_memories)} relevant memories")
        return processed_memories

    except Exception as e:
        print(f"Error searching memories: {e}")
        import traceback
        traceback.print_exc()
        return []

def search_relevant_memories_from_results(results: List, query: str) -> List[Dict[str, Any]]:
    """Helper function to process search results from a results array"""
    processed_memories = []

    for i, mem in enumerate(results):
        if isinstance(mem, dict):
            processed_memories.append(mem)
            print(f"  {i+1}. {mem.get('memory', mem.get('text', str(mem)))}")
        else:
            processed_memories.append({'memory': str(mem)})
            print(f"  {i+1}. {str(mem)}")

    return processed_memories

def generate_response_with_memory(openai_client: OpenAI, user_query: str, memories: List[Dict[str, Any]]) -> str:
    """Generate LLM response using retrieved memories as context"""

    # Format memories into context string
    if memories:
        memory_texts = []
        for mem in memories:
            # Try different possible keys for memory content
            memory_text = mem.get('memory') or mem.get('text') or mem.get('content') or str(mem)
            memory_texts.append(f"- {memory_text}")

        memory_context = "\n".join(memory_texts)
        system_prompt = f"""You are a helpful AI assistant. You have access to the following memories from previous conversations:

{memory_context}

Use this context to provide personalized and relevant responses. If the memories contain relevant information, incorporate it naturally into your response."""
    else:
        system_prompt = "You are a helpful AI assistant."

    # Create messages for OpenAI API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    print("Generating response with OpenAI...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using cost-effective model
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error generating a response."

def chat_with_memory(memory: Memory, openai_client: OpenAI, user_id: str = "user_123"):
    """Main chat function that demonstrates the full memory-enhanced conversation flow"""
    print(f"Starting chat session for user: {user_id}")
    print("=" * 60)

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print(f"Assistant: Processing your message...")

            # Step 1: Search for relevant memories
            relevant_memories = search_relevant_memories(memory, user_input, user_id)

            # Step 2: Generate response using LLM with memory context
            response = generate_response_with_memory(openai_client, user_input, relevant_memories)

            print(f"\nAssistant: {response}")

            # Step 3: Add this conversation turn to memory
            conversation = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ]
            add_conversation_to_memory(memory, conversation, user_id)

            print("-" * 40)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error in chat loop: {e}")
            continue

def demo_basic_memory_operations():
    """Demonstrate basic memory operations with sample data"""
    print("Demo: Basic Memory Operations")
    print("=" * 40)

    memory, openai_client = initialize_clients()
    user_id = "demo_user"

    # Add some initial memories
    print("\n1. Adding sample memories...")

    sample_conversations = [
        [
            {"role": "user", "content": "Hi, my name is Alice and I love pizza."},
            {"role": "assistant", "content": "Nice to meet you Alice! I'll remember that you love pizza."}
        ],
        [
            {"role": "user", "content": "I'm allergic to nuts, please remember that."},
            {"role": "assistant", "content": "I've noted your nut allergy. I'll keep that in mind for any food recommendations."}
        ],
        [
            {"role": "user", "content": "I work as a software engineer in San Francisco."},
            {"role": "assistant", "content": "Got it! You're a software engineer working in San Francisco."}
        ]
    ]

    for i, conv in enumerate(sample_conversations, 1):
        print(f"\nAdding conversation {i}/3...")
        result = add_conversation_to_memory(memory, conv, user_id)
        if "error" in result:
            print(f"Failed to add conversation {i}")
        else:
            print(f"Successfully added conversation {i}")

    # Demonstrate memory search
    print("\n2. Searching memories...")

    search_queries = [
        "What's my name?",
        "What do I do for work?",
        "Do I have any food allergies?",
        "What foods do I like?"
    ]

    for query in search_queries:
        print(f"\nQuery: '{query}'")
        memories = search_relevant_memories(memory, query, user_id, limit=2)

        if memories:
            response = generate_response_with_memory(openai_client, query, memories)
            print(f"Response: {response}")
        else:
            print("No relevant memories found.")

    print("\n" + "=" * 40)
    print("Demo completed! You can now try the interactive chat.")

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("Checking prerequisites...")

    # Check API key
    if OPENAI_API_KEY == "your-openai-api-key" or not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set properly")
        print("   Please set your OpenAI API key as an environment variable:")
        print("   export OPENAI_API_KEY='your-actual-api-key'")
        return False
    else:
        print("✓ OPENAI_API_KEY is set")

    # Check packages
    try:
        import mem0
        print("✓ mem0ai package available")
    except ImportError:
        print("❌ mem0ai package not found")
        print("   Install with: pip install mem0ai")
        return False

    try:
        import openai
        print("✓ openai package available")
    except ImportError:
        print("❌ openai package not found")
        print("   Install with: pip install openai")
        return False

    # Check write permissions
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=True)
        temp_file.close()
        print("✓ Write permissions available")
    except Exception as e:
        print(f"❌ Write permission issue: {e}")
        return False

    return True

def main():
    """Main function to run the mem0 basic example"""
    print("Mem0 Basic Example: LLM Inference with Memory")
    print("=" * 60)

    # Check prerequisites first
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return

    try:
        # Initialize clients
        memory, openai_client = initialize_clients()
        print("✓ Mem0 and OpenAI clients initialized successfully")

        # Ask user what they want to do
        print("\nChoose an option:")
        print("1. Run demo with sample data")
        print("2. Start interactive chat")
        print("3. Both (demo first, then chat)")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice in ['1', '3']:
            demo_basic_memory_operations()

        if choice in ['2', '3']:
            print("\nStarting interactive chat...")
            print("Type 'exit', 'quit', or 'bye' to end the session")
            chat_with_memory(memory, openai_client)

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your OPENAI_API_KEY environment variable is set")
        print("2. Install required packages: pip install mem0ai openai chromadb")
        print("3. Ensure you have write permissions in the current directory")
        print("4. Try running with different permissions or in a different directory")

if __name__ == "__main__":
    main()
