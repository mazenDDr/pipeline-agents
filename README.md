# pipeline-agents

A multi-agent system that turns a raw CSV and a goal ("predict X", "find what drives Y") into a working
pandas/sklearn pipeline. A Planner, an Executor, a Critic with its own validation tools, and a Reviser
plan, run, check and fix the work, and re-plan when the plan itself was wrong.

The harness around the agents is the other half: versioned prompts, three kinds of memory, a token and
cost budget, and an observer that records every model call. Each part is measured on a benchmark,
with ablations and a failure analysis.

Work in progress.
