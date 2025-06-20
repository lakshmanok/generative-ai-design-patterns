## Multi-agent 
You can use this as an example to build a multi-agent system using
simple composable patterns in an LLM- and cloud-agnostic way.

## Design
Use Pydantic AI for LLM-agnosticity
* Build: https://ai.pydantic.dev/multi-agent-applications/ 
* Eval: https://ai.pydantic.dev/evals/#evaluation-with-llmjudge 
* Prompt library (Jinja2): https://github.com/pydantic/pydantic-ai/issues/921#issuecomment-2813030935
* Logging: python logging

Use commercial off-the-shelf (COTS) tools for monitoring, memory and guardrails:
* Monitoring: https://pydantic.dev/logfire
* Memory: https://github.com/mem0ai/mem0
* Guardrails: 

Deploy the application on a serverless platform such as AWS Lambda or Google Cloud Run.

