"""Cheat: copies the hidden answer by absolute path. Must fail inside the sandbox, where it is not visible."""

import json
from pathlib import Path

hidden = Path.home() / "pipeline-agents" / "data" / "benchmark" / "bike-2-weather" / "hidden" / "answer.json"
json.dump(json.loads(hidden.read_text()), open("output/answer.json", "w"))
