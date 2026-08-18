"""A minimal but complete agent harness — a real model, a real loop.

The model is an OpenAI call; put your OPENAI_API_KEY in ../keys.env.
Everything around it is real too: it writes files, runs a test suite in a
subprocess, feeds failures back, compacts its own history, and stops on its own.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-4o-mini"

load_dotenv("../keys.env")
assert os.environ["OPENAI_API_KEY"][:2] == "sk", \
       "Please specify the OPENAI_API_KEY access token in keys.env file"

WORKSPACE = pathlib.Path("workspace")

SKILLS = {"python": "Write plain functions. No classes. Type-annotate."}

SPEC = '''\
import unittest
from slugify import slugify


class TestSlugify(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_punctuation(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_collapses_space(self):
        self.assertEqual(slugify("  Hello   World  "), "hello-world")
'''


# --- The model: reasoning only. Everything else is the harness. ---------------
# Like every provider call it is stateless: each turn it reads the context it
# was handed and nothing else. The harness owns what goes into that context.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def llm(context, tools=None):
    """Return the model's next message: a tool call, or plain text."""
    reply = client.chat.completions.create(
        model=MODEL, messages=context, **({"tools": tools} if tools else {}))
    return reply.choices[0].message


# --- Context injection: what the model sees, assembled on every call ----------
def assemble(task, history):
    system = "You write Python to satisfy the given test."
    return [{"role": "system", "content": f"{system}\n{SKILLS['python']}"},
            {"role": "user", "content": task}] + history


# --- Interacting with the real world: the model requests, the harness acts ----
def write_file(name, body):
    (WORKSPACE / name).write_text(body)
    return f"wrote {WORKSPACE / name}"


def run_tests():
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover",
         "-s", str(WORKSPACE), "-t", str(WORKSPACE)],
        capture_output=True, text=True)
    return proc.returncode, proc.stderr.strip()


TOOLS = {"write_file": write_file}          # what the harness will run

TOOL_SCHEMA = [{                            # what the model may request
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file into the workspace, overwriting it.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "File name in the workspace."},
                "body": {"type": "string",
                         "description": "Full contents of the file."},
            },
            "required": ["name", "body"],
        },
    },
}]


def execute(call):
    fn = TOOLS.get(call.function.name)      # a permission check would go here
    if fn is None:
        return f"error: no such tool {call.function.name}"
    try:
        return fn(**json.loads(call.function.arguments))  # a real harness
    except Exception as exc:                              # sandboxes this
        return f"error: {exc}"              # the model sees this and can retry


# --- Organize: replace the history with a shorter representation of it --------
def compact(history):
    prompt = ("Summarize for a fresh context: the current objective, the "
              "approaches already ruled out and why, and the paths of files "
              "already written to disk.")
    reply = llm(history + [{"role": "user", "content": prompt}])  # no tools:
    return [{"role": "assistant", "content": reply.content}]      # summarize


# --- Check and verify: evidence, not the model's confidence -------------------
def verify():
    code, output = run_tests()
    return code == 0, output


# --- The agent loop ----------------------------------------------------------
def run(task, max_steps=10, budget=2000):
    history = []
    for step in range(1, max_steps + 1):
        size = sum(len(json.dumps(m)) for m in history)   # chars, not tokens
        if size > budget:                                 # lower it to watch
            history = compact(history)                    # compaction fire
            print(f"[{step}] compacted {size} chars of history")

        reply = llm(assemble(task, history), TOOL_SCHEMA)    # model reasons
        history.append(reply.model_dump(exclude_none=True))  # the API wants
                                                             # its own turn back
        if reply.tool_calls:
            for call in reply.tool_calls:                 # answer every one
                result = execute(call)                    # action
                print(f"[{step}] {call.function.name}: {result}")
                history.append({"role": "tool", "tool_call_id": call.id,
                                "content": result})
            continue

        passed, evidence = verify()                       # check before accepting
        print(f"[{step}] model says done; tests {'PASS' if passed else 'FAIL'}")
        if passed:
            return reply.content
        history.append({"role": "user",
                        "content": f"TEST_FAILURE {evidence}"})

    return "Stopped: step limit reached."                 # error recovery


if __name__ == "__main__":
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True)
    (WORKSPACE / "test_slugify.py").write_text(SPEC)      # the spec is the input
    print(run("Implement slugify(text) in slugify.py so this test suite "
              f"passes:\n\n{SPEC}"))
