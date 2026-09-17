"""Benchmark families. The dev/test split is by family, so no dataset is seen on both sides."""

from pipeline_agents.bench.families import adult, air, bank, bike, credit, diabetes, retail

FAMILIES = {
    "bike": bike,
    "adult": adult,
    "air": air,  # dev
    "bank": bank,
    "diabetes": diabetes,
    "credit": credit,
    "retail": retail,  # test
}
TASKS = [task for module in FAMILIES.values() for task in module.TASKS]
