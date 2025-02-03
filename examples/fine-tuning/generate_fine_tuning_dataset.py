import json
import csv
import random
from typing import List, Dict

NUM_EXAMPLES = 200
SYSTEM_PROMPT = "You are a helpful assistant converting notes to professional emails."

# Templates for different types of emails
TEMPLATES = {
    "meeting": {
        "notes": [
            """mtg w/ {team} team @ {location} - {date} {time}
disscussed {topic}
key points:
- {point1}
- {point2}
- {point3}
next:
- {next1}
- {next2}
pls respond w/ questions""",
            """sync w/ {team} @ {location}
date/time: {date} {time}
talked abt {topic}
covered:
{point1}
{point2}
{point3}
todo:
{next1}
{next2}
lmk if u need clarification""",
            """quick {team} update in {location}
{date} {time}
topic: {topic}
main points:
{point1}
{point2}
{point3}
actions:
{next1}
{next2}
ping me 4 questions""",
        ],
        "subject": "{team} Team Meeting Summary - {date}",
        "body": """Dear {team} team,

I hope this email finds you well. I'm writing to summarize our team meeting that took place on {date} at {time} in {location}.

During our discussion about {topic}, we covered several key points:
{bullet_points}

Next steps:
{next_steps}

Please let me know if you have any questions or if I missed anything important.

Best regards,
{name}"""
    },
    "request": {
        "notes": [
            """urgent: need {item} by {deadline}
for stakeholder presentaton
details:
- {context}
- will need ur help asap
- send 2 me when rdy""",
            """{item} request asap
deadline: {deadline}
context: {context}
need this 4 upcoming meeting
ping me when done thx""",
            """need {item} urgent!!!
must have by: {deadline}
why: {context}
lemme know if u need more info
appreciate ur help""",
        ],
        "subject": "Request for {item}",
        "body": """Hi {name},

I hope you're doing well. I'm reaching out because I need {item} by {deadline}.

{context}

Could you please help me with this? Let me know if you need any additional information.

Thank you in advance for your help.

Best,
{name}"""
    },
    "feedback": {
        "notes": [
            """feedback 4 {project}
looked @ everything, here r my thots:
{point1}
{point2}
{point3}
lets discuss if needed""",
            """{project} review notes
main points 2 improve:
- {point1}
- {point2}
- {point3}
lmk if u wanna chat more abt it""",
            """thoughts on {project}:
found sum issues:
{point1}
{point2}
{point3}
can explain in detail if needed""",
        ],
        "subject": "Feedback on {project}",
        "body": """Hello {name},

Thank you for sharing {project} with me. I've reviewed it and would like to provide some constructive feedback.

Here are my main observations:
{points}

I believe implementing these suggestions would further strengthen the {project}. Please let me know if you'd like to discuss any of these points in more detail.

Kind regards,
{name}"""
    }
}

# Sample data for generation
DATA = {
    "teams": ["Marketing", "Engineering", "Sales", "Product", "Design", "HR", "Finance"],
    "topics": ["Q2 roadmap", "project timeline", "budget review", "team priorities", "resource allocation"],
    "items": ["quarterly report", "project proposal", "budget forecast", "meeting minutes", "presentation deck"],
    "projects": ["website redesign", "mobile app", "marketing campaign", "sales strategy", "customer survey"],
    "bullet_points": [
        "Reviewed current progress and milestones",
        "Discussed challenges and potential solutions",
        "Aligned on priorities for the next quarter",
        "Identified resource needs and constraints",
        "Updated timeline and deliverables"
    ],
    "next_steps": [
        "Schedule follow-up meeting next week",
        "Share updated documentation by Friday",
        "Create action items list",
        "Set up individual check-ins",
        "Review progress next sprint"
    ]
}

def format_bullet_points(points: List[str]) -> str:
    """Format a list of points as bullet points."""
    return "\n".join(f"- {point}" for point in points)

def generate_entry(entry_type: str = None) -> Dict[str, str]:
    """Generate a single note-email pair."""
    if not entry_type:
        entry_type = random.choice(list(TEMPLATES.keys()))

    template = TEMPLATES[entry_type]

    # Select bullet points and next steps
    selected_points = random.sample(DATA["bullet_points"], 3)
    selected_next_steps = random.sample(DATA["next_steps"], 2)

    # Generate variables for template with placeholders
    vars = {
        "team": random.choice(DATA["teams"]),
        "date": "{date}",
        "time": "{time}",
        "location": "{location}",
        "topic": random.choice(DATA["topics"]),
        "item": random.choice(DATA["items"]),
        "deadline": "{deadline}",
        "project": random.choice(DATA["projects"]),
        "name": "{name}",
        "context": "This is needed for our upcoming presentation to stakeholders.",
        "point1": selected_points[0],
        "point2": selected_points[1],
        "point3": selected_points[2],
        "next1": selected_next_steps[0],
        "next2": selected_next_steps[1],
        "points": "\n".join(f"- {point}" for point in selected_points),
        "bullet_points": format_bullet_points(selected_points),
        "next_steps": format_bullet_points(selected_next_steps)
    }

    # Generate note, subject, and body using templates
    note = random.choice(template["notes"]).format(**vars)
    subject = template["subject"].format(**vars)
    body = template["body"].format(**vars)

    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": note + " Style: " + entry_type},
        {"role": "assistant", "content": "Subject: " + subject + "\n\nBody: " + body}
    ]}


def generate_dataset(size: int = 100) -> List[Dict[str, str]]:
    """Generate a dataset of the specified size."""
    return [generate_entry() for _ in range(size)]

def save_dataset(dataset: List[Dict[str, str]], filename: str = None) -> None:
    """Save the dataset in the specified format."""
    if not filename:
        filename = "training_data.jsonl"

    with open(filename, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')


def main():
    """Generate example datasets in different formats."""
    # Generate a small example dataset
    print("Generating example dataset...")
    example_dataset = generate_dataset(5)
    print("\nExample entries:")
    print(json.dumps(example_dataset[:2], indent=2))

    # Generate and save full dataset in different formats
    print("\nGenerating full dataset...")
    full_dataset = generate_dataset(NUM_EXAMPLES)

    save_dataset(full_dataset)
    print(f"\nDataset saved in jsonl format.")

if __name__ == "__main__":
    main()
