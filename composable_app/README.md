## Composable Patterns Application Architecture
You can use this as an example to build a multi-agent system using
simple composable patterns in an LLM- and cloud-agnostic way and
using primarily OSS components.

## Design
Use Pydantic AI for LLM-agnosticity
* Build: https://ai.pydantic.dev/multi-agent-applications/
* Prompt management (Jinja2): https://github.com/pydantic/pydantic-ai/issues/921#issuecomment-2813030935
* Logging: https://docs.python.org/3/library/logging.html
* Eval: https://ai.pydantic.dev/evals/#evaluation-with-llmjudge

We build these services using LLM-as-Judge, but log inputs and outputs so as to post-train a SLM later
* Guardrails: See utils/guardrails.py
* 

Use commercial off-the-shelf (COTS) tools for monitoring, memory and optionally for guardrails and evaluation:
* Monitoring: https://pydantic.dev/logfire
* Memory: https://github.com/mem0ai/mem0
* Guardrails: https://github.com/guardrails-ai/guardrails (optional: Toxicity, etc. as second layer)
 

## How to run it locally
Edit keys.env and add your Gemini API key to it (you don't need the others unless you plan to change LLMs):
```
GEMINI_API_KEY=AI...
```

Install the packages:
```
pip install -r requirements.txt 
```

Try out the command-line app:
``` 
python cmdline_app.py 
```

Suggested topics:
* Battle of the Bulge
* Solve: x=3 = 5

Try out the GUI interface:
``` 
streamlit run streamlit_app.py 
```

Check out the logs, configured in logging.json to save only the prompt texts:
``` cat app.log ```

## Deploy application
This is a Dockerized application; you can deploy it on
a serverless platform such as AWS Farsight or Google Cloud Run.


## How it works
The workflow of this application is depicted below:
![k12_workflow](k12_content_writing_workflow.png)

All the prompts are in the prompts directory.
Look at the prompts and correlate them with the diagram above.

The code itself involves hooking up the workflow and using the right data structures.

