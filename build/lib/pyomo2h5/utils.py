import re
import numpy as np
from ruamel.yaml.scalarfloat import ScalarFloat


def safe_encode(value, default="") -> bytes:
    return str(value if value is not None else default).encode("ascii", "ignore")


def extract_unit(doc: str) -> str:
    match = re.search(r"\(([^)]+)\)", doc)
    return match.group(1) if match else ""


def parse_component_path(name: str) -> tuple[str | None, list[str]]:
    match = re.match(r"scenario\[(?:'([^']+)'|([^]]+))\]\.(.+)", name)
    if match:
        scenario = match.group(1) or match.group(2)
        remaining = match.group(3)
        parts = re.split(r"\.|\[|\]", remaining)
        return scenario, [p for p in parts if p]
    return None, [name]


def convert_scalarfloats(obj):
    if isinstance(obj, ScalarFloat):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_scalarfloats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_scalarfloats(v) for v in obj]
    return obj


def replace_none(obj):
    if obj is None:
        return "None"
    elif isinstance(obj, dict):
        return {k: replace_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_none(v) for v in obj]
    return obj
