"""Benchmark families. The dev/test split is by family, so no dataset is seen on both sides."""

from pipeline_agents.bench.families import bike

FAMILIES = {"bike": bike}
TASKS = [task for module in FAMILIES.values() for task in module.TASKS]
