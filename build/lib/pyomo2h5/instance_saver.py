import pyomo.environ as pyo
import numpy as np
from .utils import parse_component_path, safe_encode, extract_unit


class InstanceSaver:
    def _get_or_create_group(self, base_group, path_parts):
        group = base_group
        for part in path_parts:
            group = group.require_group(part)
        return group

    def _add_metadata(self, dataset, component, component_type, path_parts):
        unit = (
            extract_unit(getattr(component, "doc", ""))
            if component_type == pyo.Param
            else ""
        )
        dataset.attrs["unit"] = safe_encode(unit)
        dataset.attrs["description"] = safe_encode(getattr(component, "doc", ""))

    def _save_constraints(self, instance, file):
        group = file.require_group("Constraint")
        for comp in instance.component_objects(pyo.Constraint, active=True):
            try:
                name = str(comp.name)
                scenario, path_parts = parse_component_path(name)
                indices = list(comp.keys())

                dtype = [
                    ("body", "S1000"),
                    ("value", "f8"),
                    ("lower_bound", "f8"),
                    ("upper_bound", "f8"),
                ]
                array = np.zeros(len(indices), dtype=dtype)

                for i, idx in enumerate(indices):
                    c = comp[idx]
                    array["body"][i] = str(c.body).encode("ascii", "ignore")
                    array["value"][i] = pyo.value(c.body)
                    array["lower_bound"][i] = (
                        pyo.value(c.lower) if c.has_lb() else np.nan
                    )
                    array["upper_bound"][i] = (
                        pyo.value(c.upper) if c.has_ub() else np.nan
                    )

                target = (
                    group.require_group("Scenario").require_group(scenario)
                    if scenario
                    else group
                )
                current_group = self._get_or_create_group(target, path_parts[:-1])
                dataset_name = path_parts[-1]

                if dataset_name in current_group:
                    del current_group[dataset_name]

                dset = current_group.create_dataset(dataset_name, data=array)
                dset.attrs["description"] = safe_encode(getattr(comp, "doc", ""))
            except Exception as e:
                print(f"Error saving constraint {name}: {e}")
