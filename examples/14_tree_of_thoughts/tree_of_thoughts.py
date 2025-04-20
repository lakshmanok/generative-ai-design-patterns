import anthropic
import os
import json
import heapq
import time
from dotenv import load_dotenv
from typing import List, Dict, Any


load_dotenv(dotenv_path="examples/keys.env", verbose=True, override=True)

class ClaudeTreeOfThoughts:
    """
    Tree of Thoughts implementation using Claude API for both thought generation and evaluation.
    """

    def __init__(self,
                 api_key: str,
                 num_thoughts_per_step: int = 3,
                 max_steps: int = 5,
                 beam_width: int = 3,
                 model: str = "claude-3-7-sonnet-20250219",
                 verbose: bool = False):
        """
        Initialize the Tree of Thoughts with Claude.

        Args:
            api_key: Anthropic API key
            num_thoughts_per_step: Number of thoughts to generate at each step
            max_steps: Maximum number of reasoning steps to take
            beam_width: Width of the beam for beam search
            model: Claude model to use
            verbose: Whether to print detailed progress
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.num_thoughts_per_step = num_thoughts_per_step
        self.max_steps = max_steps
        self.beam_width = beam_width
        self.model = model
        self.verbose = verbose
        self.call_count = 0

    def generate_thoughts(self, state: str, step: int) -> List[str]:
        """
        Generate multiple possible next thoughts using Claude.

        Args:
            state: Current reasoning state
            step: Current step number

        Returns:
            List of generated thoughts
        """
        self.call_count += 1

        prompt = f"""
        {state}

        You are solving a problem step-by-step using the Tree of Thoughts method.
        Think about the problem state above and generate {self.num_thoughts_per_step} distinct and diverse next steps.

        This is step {step} of up to {self.max_steps} steps.

        Generate {self.num_thoughts_per_step} different possible next thoughts to make progress on this problem.
        Make each thought meaningfully different to explore diverse approaches.

        Format your response as a JSON array of strings, with each string being a single thought:
        ["Thought 1...", "Thought 2...", "Thought 3..."]

        Only provide the JSON array, nothing else.
        """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.8,  # Higher temperature for diverse thoughts
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract JSON array from response
            content = response.content[0].text
            # Clean the response to ensure it's valid JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            thoughts = json.loads(content)

            # Ensure we have exactly the right number of thoughts
            thoughts = thoughts[:self.num_thoughts_per_step]

            if self.verbose:
                print(f"\nGenerated {len(thoughts)} thoughts for step {step}:")
                for i, thought in enumerate(thoughts):
                    print(f"  {i+1}. {thought}")

            return thoughts

        except Exception as e:
            print(f"Error generating thoughts: {e}")
            # Return a fallback thought if API call fails
            return [f"Let me reconsider the problem from a different angle."]

    def evaluate_state(self, state: str, problem: str) -> float:
        """
        Evaluate the promise of a reasoning path using Claude.

        Args:
            state: Current reasoning state to evaluate
            problem: Original problem statement

        Returns:
            Score between 0 and 1 indicating the promise of this reasoning path
        """
        self.call_count += 1

        prompt = f"""
        Problem: {problem}

        Reasoning path:
        {state}

        On a scale from 0 to 100, evaluate how promising this reasoning path is for solving the problem.
        Consider:
        1. Correctness - Is the reasoning logically sound?
        2. Progress - How much progress has been made toward the solution?
        3. Insight - Does the reasoning show understanding of the key aspects?
        4. Potential - How likely is this path to lead to a complete solution?

        Respond with a single integer score between 0 and 100. Higher scores indicate more promising paths.
        Only provide the number, nothing else.
        """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0.2,  # Lower temperature for consistent evaluation
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract score from response
            content = response.content[0].text.strip()
            score = int(content) / 100.0  # Convert to 0-1 scale

            if self.verbose:
                print(f"Evaluated state (score: {score:.2f})")

            return score

        except Exception as e:
            print(f"Error evaluating state: {e}")
            # Return a middle score if API call fails
            return 0.5

    def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve a problem using Tree of Thoughts with Claude.

        Args:
            problem: Problem statement to solve

        Returns:
            Dictionary containing the solution and reasoning path
        """
        start_time = time.time()
        self.call_count = 0

        # Initialize with the problem statement
        initial_state = f"Problem: {problem}\n\nInitial thoughts:"

        # Priority queue for beam search [(score, state, reasoning_path, step)]
        beam = [(0.5, initial_state, [], 0)]  # Start with a neutral score of 0.5

        best_final_states = []

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step} ---")

            # Store all candidates for this step
            candidates = []

            # Process each state in the current beam
            for score, current_state, reasoning_path, current_step in beam:
                if current_step >= step:
                    # This state is from a future step, keep it in the beam
                    candidates.append((score, current_state, reasoning_path, current_step))
                    continue

                # Generate thoughts from the current state
                thoughts = self.generate_thoughts(current_state, step)

                # Process each generated thought
                for thought in thoughts:
                    # Create new state by appending the thought
                    new_state = f"{current_state}\nStep {step}: {thought}"

                    # Create a new reasoning path
                    new_path = reasoning_path + [f"Step {step}: {thought}"]

                    # Evaluate the promise of this new state
                    new_score = self.evaluate_state(new_state, problem)

                    # Add to candidates (negate score for max-heap behavior with min-heap)
                    candidates.append((-new_score, new_state, new_path, step))

                    # Check if we've found a great solution
                    if new_score > 0.9:
                        best_final_states.append((new_score, new_state, new_path))

            # If we have promising complete solutions, we can stop early
            if best_final_states and best_final_states[0][0] > 0.95:
                break

            # Select top-k candidates for the next beam
            beam = []
            for candidate in heapq.nsmallest(self.beam_width, candidates):
                score, state, path, s = candidate
                beam.append((-score, state, path, s))  # Convert score back to positive

            if self.verbose:
                print(f"\nTop {len(beam)} states after step {step}:")
                for i, (score, state, _, _) in enumerate(beam):
                    state_preview = state.split('\n')[-1] if '\n' in state else state
                    print(f"  {i+1}. Score: {score:.2f} | {state_preview[:60]}...")

        # Find best final state
        if best_final_states:
            best_score, best_state, best_path = max(best_final_states, key=lambda x: x[0])
        else:
            # If no final states were marked as excellent, use the best from the beam
            best_score, best_state, best_path, _ = max(beam, key=lambda x: x[0])

        # Generate a concise solution summary
        summary = self.generate_solution_summary(problem, best_state)

        elapsed_time = time.time() - start_time

        return {
            "problem": problem,
            "solution_score": best_score,
            "reasoning_path": best_path,
            "final_state": best_state,
            "solution_summary": summary,
            "stats": {
                "api_calls": self.call_count,
                "elapsed_time": elapsed_time,
                "steps_taken": len(best_path)
            }
        }

    def generate_solution_summary(self, problem: str, final_state: str) -> str:
        """
        Generate a concise summary of the solution using Claude.

        Args:
            problem: Original problem statement
            final_state: Final reasoning state

        Returns:
            A concise summary of the solution
        """
        self.call_count += 1

        prompt = f"""
        Problem: {problem}

        Complete reasoning path:
        {final_state}

        Please provide a concise summary of the solution to this problem based on the reasoning path above.
        Focus on the key insights and the answer to the original problem.
        """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text.strip()

        except Exception as e:
            print(f"Error generating solution summary: {e}")
            return "Failed to generate solution summary due to an error."

# Example usage for a complex reasoning problem
def solve_tot(problem):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")
    # print(f"api_key: {api_key}")
    if not api_key:
        raise ValueError("Anthropic API key not found in environment variables")

    tot = ClaudeTreeOfThoughts(
        api_key=api_key,
        num_thoughts_per_step=3,
        max_steps=4,
        beam_width=3,
        verbose=True
    )

    solution = tot.solve(problem)

    print("\n=== SOLUTION ===")
    print(f"Problem: {solution['problem']}")
    print(f"Final Score: {solution['solution_score']:.2f}")
    print("\nReasoning Path:")
    for step in solution['reasoning_path']:
        print(f"  {step}")
    print("\nSolution Summary:")
    print(solution['solution_summary'])
    print("\nStats:")
    print(f"  API Calls: {solution['stats']['api_calls']}")
    print(f"  Time Taken: {solution['stats']['elapsed_time']:.2f} seconds")
    print(f"  Steps Taken: {solution['stats']['steps_taken']}")
    
    
def main():
    solve_tot("""
I need to optimize our supply chain using Tree of Thoughts:

Current situation:
- 3 potential manufacturing locations (Mexico, Vietnam, Poland)
- 4 distribution centers (Atlanta, Chicago, Dallas, Seattle)
- 2 primary shipping methods (air, sea)
- Historical demand fluctuations of ±20%
- Recent disruptions in Asian shipping routes

For each possible configuration:
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
    """)

if __name__ == "__main__":
    main()
