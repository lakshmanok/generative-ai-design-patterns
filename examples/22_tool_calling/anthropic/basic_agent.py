import os
import json
from typing import List, Dict, Any, Callable
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Claude client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

class Tool:
    """A tool that the agent can use to interact with the world."""

    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function

    def execute(self, **kwargs) -> Any:
        """Execute the tool with the provided arguments."""
        return self.function(**kwargs)

    def to_dict(self) -> Dict[str, str]:
        """Convert tool to a dictionary for Claude's context."""
        return {
            "name": self.name,
            "description": self.description
        }


class Task:
    """A task that the agent needs to complete."""

    def __init__(self, id: str, description: str, completed: bool = False,
                 dependencies: List[str] = None, result: Any = None):
        self.id = id
        self.description = description
        self.completed = completed
        self.dependencies = dependencies or []
        self.result = result

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to a dictionary for Claude's context."""
        return {
            "id": self.id,
            "description": self.description,
            "completed": self.completed,
            "dependencies": self.dependencies,
            "result": str(self.result) if self.result is not None else None
        }


class Agent:
    """A Claude-powered agent that can use tools to accomplish tasks."""

    def __init__(self, tools: List[Tool] = None):
        self.tools = tools or []
        self.tasks: List[Task] = []
        self.memory: List[Dict[str, Any]] = []

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)

    def add_task(self, task: Task) -> None:
        """Add a task to the agent's task list."""
        self.tasks.append(task)

    def get_available_tasks(self) -> List[Task]:
        """Get tasks that are ready to be executed (all dependencies are completed)."""
        available_tasks = []
        completed_task_ids = [task.id for task in self.tasks if task.completed]

        for task in self.tasks:
            if not task.completed and all(dep in completed_task_ids for dep in task.dependencies):
                available_tasks.append(task)

        return available_tasks

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name with the provided arguments."""
        for tool in self.tools:
            if tool.name == tool_name:
                result = tool.execute(**kwargs)
                # Add to memory
                self.memory.append({
                    "type": "tool_execution",
                    "tool": tool_name,
                    "args": kwargs,
                    "result": result
                })
                return result

        raise ValueError(f"Tool '{tool_name}' not found")

    def mark_task_complete(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed and store its result."""
        for task in self.tasks:
            if task.id == task_id:
                task.completed = True
                task.result = result
                # Add to memory
                self.memory.append({
                    "type": "task_completion",
                    "task_id": task_id,
                    "result": result
                })
                return

        raise ValueError(f"Task '{task_id}' not found")

    def create_claude_prompt(self, task: Task) -> str:
        """Create a prompt for Claude based on the current state."""
        tools_descriptions = [tool.to_dict() for tool in self.tools]
        tasks_descriptions = [t.to_dict() for t in self.tasks]

        # Format memory/history for context
        memory_str = json.dumps(self.memory, indent=2)

        prompt = f"""
You are an AI assistant tasked with helping complete tasks. You have access to the following tools:

{json.dumps(tools_descriptions, indent=2)}

Here are all the tasks (both completed and pending):

{json.dumps(tasks_descriptions, indent=2)}

Your memory/execution history:
{memory_str}

Your current task is:
ID: {task.id}
Description: {task.description}

You need to decide:
1. If you need to use a tool to complete this task, specify which tool and with what parameters.
2. If the task can be completed with your knowledge, provide the answer directly.
3. If the task depends on other incomplete tasks, explain why you can't complete it yet.

Respond in JSON format with "reasoning", "action" (either "use_tool", "complete_task", or "cannot_complete"), and additional fields depending on the action:
- For "use_tool": Include "tool_name" and "tool_parameters" (a dictionary of parameter names and values)
- For "complete_task": Include "result" with your answer
- For "cannot_complete": Include "reason" explaining why
"""

        prompt += """
Example response:
```json
{
  "reasoning": "I need to search for information about X to complete this task.",
  "action": "use_tool",
  "tool_name": "search",
  "tool_parameters": {"query": "information about X"}
}
```

Or:

```json
{
  "reasoning": "I know the answer to this question from my training.",
  "action": "complete_task",
  "result": "The answer is Y because..."
}
```
"""
        return prompt

    def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a single task using Claude."""
        prompt = self.create_claude_prompt(task)

        # Call Claude API
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response = message.content[0].text

        # Extract JSON from response
        try:
            # Find JSON between ```json and ``` if it exists
            if "```json" in response and "```" in response.split("```json")[1]:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                # Otherwise try to parse the entire response as JSON
                json_str = response

            action_data = json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, create a fallback response
            action_data = {
                "reasoning": "Failed to parse Claude's response as JSON",
                "action": "cannot_complete",
                "reason": "Technical error in response parsing"
            }

        # Add to memory
        self.memory.append({
            "type": "claude_response",
            "task_id": task.id,
            "prompt": prompt,
            "response": response,
            "parsed_action": action_data
        })

        return action_data

    def execute_task(self, task: Task) -> None:
        """Execute a single task."""
        action_data = self.process_task(task)

        if action_data["action"] == "use_tool":
            tool_name = action_data["tool_name"]
            tool_parameters = action_data["tool_parameters"]

            result = self.execute_tool(tool_name, **tool_parameters)

            # Call Claude again with the tool result
            self.memory.append({
                "type": "follow_up",
                "task_id": task.id,
                "tool_result": result
            })

            # Recursive call to continue processing with new information
            self.execute_task(task)

        elif action_data["action"] == "complete_task":
            self.mark_task_complete(task.id, action_data["result"])

        elif action_data["action"] == "cannot_complete":
            # Task remains incomplete
            pass

    def run(self) -> None:
        """Run the agent until all tasks are completed or no progress can be made."""
        progress_made = True

        while progress_made:
            progress_made = False
            available_tasks = self.get_available_tasks()

            if not available_tasks:
                break

            for task in available_tasks:
                initial_completed_count = sum(1 for t in self.tasks if t.completed)
                self.execute_task(task)
                new_completed_count = sum(1 for t in self.tasks if t.completed)

                if new_completed_count > initial_completed_count:
                    progress_made = True

        # Print final status
        completed_tasks = sum(1 for task in self.tasks if task.completed)
        total_tasks = len(self.tasks)
        print(f"Completed {completed_tasks}/{total_tasks} tasks")

        for task in self.tasks:
            status = "✓" if task.completed else "✗"
            print(f"{status} {task.id}: {task.description}")
            if task.completed and task.result:
                print(f"  Result: {task.result}")
            print()

# Example tools

def search_tool(query: str) -> str:
    """Simulated search tool that returns fake results for any query."""
    return f"Search results for '{query}': This is simulated search data for demonstration purposes."

def calculator_tool(expression: str) -> float:
    """A simple calculator tool that evaluates mathematical expressions."""
    try:
        # Warning: eval can be dangerous in production code
        # Use a safer alternative in real applications
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"

def weather_tool(location: str) -> str:
    """Simulated weather tool that returns fake weather data."""
    return f"Weather in {location}: 72°F, Partly Cloudy"

# Example usage

def main():
    # Create agent with tools
    agent = Agent()

    # Add tools
    agent.add_tool(Tool("search", "Search for information on the web", search_tool))
    agent.add_tool(Tool("calculator", "Evaluate mathematical expressions", calculator_tool))
    agent.add_tool(Tool("weather", "Get weather information for a location", weather_tool))

    # Add tasks
    agent.add_task(Task("task_1", "What is the capital of Portugal?"))
    agent.add_task(Task("task_2", "Calculate 1234 * 5678", dependencies=["task_1"]))
    agent.add_task(Task("task_3", "What's the weather like in Lisbon?", dependencies=["task_1"]))
    agent.add_task(Task("task_4", "Summarize all the information collected", dependencies=["task_2", "task_3"]))

    # Run the agent
    agent.run()

if __name__ == "__main__":
    main()
