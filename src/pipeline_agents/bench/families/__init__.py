"""Benchmark families. The dev/test split is by family, so no dataset is seen on both sides."""

from pipeline_agents.bench.families import adult, air, bank, bike, credit, diabetes

FAMILIES = {"bike": bike, "adult": adult, "air": air, "bank": bank, "diabetes": diabetes, "credit": credit}
TASKS = [task for module in FAMILIES.values() for task in module.TASKS]
