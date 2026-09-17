"""Benchmark families. The dev/test split is by family, so no dataset is seen on both sides."""

from pipeline_agents.bench.families import adult, air, bank, bike, diabetes

FAMILIES = {"bike": bike, "adult": adult, "air": air, "bank": bank, "diabetes": diabetes}
TASKS = [task for module in FAMILIES.values() for task in module.TASKS]
