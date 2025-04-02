import pyomo.environ as pyo
import numpy as np
from functools import reduce
import re
from collections.abc import Mapping
from .utils import (
    parse_component_path,
    safe_encode,
    extract_unit,
    replace_none,
    convert_scalarfloats,
)


class InstanceSaver:
    def _get_or_create_group(self, base_group, path_parts):
        group = base_group
        for part in path_parts:
            group = group.require_group(part)
        return group

    def _add_metadata(self, dataset, component, component_type):
        unit = (
            extract_unit(getattr(component, "doc", ""))
            if component_type == pyo.Param
            else ""
        )
        dataset.attrs["unit"] = safe_encode(unit)
        dataset.attrs["description"] = safe_encode(getattr(component, "doc", ""))

    def _save_objective(self, instance, group=None):
        base_group = group if group is not None else self.file
        obj_group = base_group.require_group("Objective")

        for obj in instance.component_objects(pyo.Objective, active=True):
            try:
                name = str(obj.name)
                expr = obj.expr
                value = pyo.value(expr)
                expr_str = str(expr)
                sense = "minimize" if obj.sense == pyo.minimize else "maximize"

                # Define structured dtype
                dtype = [("expression", f"S{len(expr_str) + 10}"), ("value", "f8")]
                array = np.zeros(1, dtype=dtype)
                array["expression"][0] = expr_str.encode("ascii", "ignore")
                array["value"][0] = value

                if name in obj_group:
                    del obj_group[name]

                dset = obj_group.create_dataset(name, data=array)
                dset.attrs["sense"] = sense
                dset.attrs["description"] = safe_encode(getattr(obj, "doc", ""))

            except Exception as e:
                print(f"Error saving Objective {name}: {e}")

    def _save_components(self, instance, component_type, group_name, group=None):
        base_group = group if group is not None else self.file
        component_group = base_group.require_group(group_name)

        for component in instance.component_objects(component_type, active=True):
            try:
                comp_name = str(component.name)
                scenario, path_parts = parse_component_path(comp_name)

                dtype, combined_header = self._build_dtype(component, component_type)
                structured_array = np.zeros(len(set(component.keys())), dtype=dtype)
                self._fill_structured_array(
                    structured_array, component, component_type, combined_header
                )

                if scenario:
                    scenario_group = component_group.require_group("Scenario")
                    current_group = self._get_or_create_group(
                        scenario_group.require_group(scenario), path_parts[:-1]
                    )
                else:
                    current_group = self._get_or_create_group(
                        component_group, path_parts[:-1]
                    )

                dataset_name = path_parts[-1]
                if dataset_name in current_group:
                    del current_group[dataset_name]

                dataset = current_group.create_dataset(
                    dataset_name, data=structured_array
                )
                self._add_metadata(dataset, component, component_type)

            except Exception as e:
                print(f"Error saving component {comp_name}: {e}")

    def _serialize_constraints_to_dataset(
        self, constraints, group, dataset_name, description=""
    ):
        """
        constraints: dict[str, pyo.Constraint] or iterable of (name, pyo.Constraint)
        group: h5py group
        dataset_name: str
        description: str
        """
        dtype = [
            ("body", "S1000"),
            ("value", "f8"),
            ("lower_bound", "f8"),
            ("upper_bound", "f8"),
        ]

        array = np.zeros(len(constraints), dtype=dtype)

        for i, (name, con) in enumerate(constraints.items()):
            array["body"][i] = str(con.body).encode("ascii", "ignore")
            array["value"][i] = pyo.value(con.body)
            array["lower_bound"][i] = pyo.value(con.lower) if con.has_lb() else np.nan
            array["upper_bound"][i] = pyo.value(con.upper) if con.has_ub() else np.nan

        if dataset_name in group:
            del group[dataset_name]

        dset = group.create_dataset(dataset_name, data=array)
        dset.attrs["description"] = safe_encode(description or "")

    def _save_constraints(self, instance, group=None):
        base_group = group if group is not None else self.file
        group = base_group.require_group("Constraint")
        for comp in instance.component_objects(pyo.Constraint, active=True):
            try:
                name = str(comp.name)
                scenario, path_parts = parse_component_path(name)
                indices = list(comp.keys())
                constraints = {str(idx): comp[idx] for idx in indices}

                target = (
                    group.require_group("Scenario").require_group(scenario)
                    if scenario
                    else group
                )
                current_group = self._get_or_create_group(target, path_parts[:-1])
                dataset_name = path_parts[-1]

                self._serialize_constraints_to_dataset(
                    constraints,
                    current_group,
                    dataset_name,
                    description=getattr(comp, "doc", ""),
                )
            except Exception as e:
                print(f"Error saving constraint {name}: {e}")

    def _save_model_sets(self, instance, group=None):
        base_group = group if group is not None else self.file
        component_group = base_group.require_group("Set")

        # First handle regular Sets
        for s in instance.component_data_objects(pyo.Set, active=True):
            try:
                keys = self._split_index_name(s.name)
                doc = getattr(s, "doc", "")

                value = list(s.data())

                self._save_set_dataset(component_group, keys, value, doc)
            except Exception as e:
                print(f"Error saving Set {s.name}: {e}")

        # Now handle RangeSets separately (they are components, not data objects)
        for s in instance.component_objects(pyo.RangeSet, active=True):
            try:
                keys = self._split_index_name(s.name)
                doc = getattr(s, "doc", "")

                # Ensure it's constructed
                if not s.is_constructed():
                    s.construct()

                value = list(s)

                self._save_set_dataset(component_group, keys, value, doc)
            except Exception as e:
                print(f"Error saving RangeSet {s.name}: {e}")

    def _save_set_dataset(self, base_group, keys, value, doc):
        # Convert to ASCII array
        if value:
            max_len = max(len(str(v)) for v in value)
            data_array = np.array(
                [str(v).encode("ascii", "ignore") for v in value], dtype=f"S{max_len}"
            )
        else:
            data_array = np.array([], dtype="S1")

        # Navigate to group
        current_group = self._get_or_create_group(base_group, keys[:-1])
        dataset_name = keys[-1]

        if dataset_name in current_group:
            del current_group[dataset_name]

        dset = current_group.create_dataset(dataset_name, data=data_array)
        dset.attrs["description"] = safe_encode(doc)
        dset.attrs["unit"] = safe_encode("")

    def _split_index_name(self, name):
        parts = re.split(r"\[|\]|(?<!\d)\.(?!\d)", name)
        return [x for x in parts if x]

    def _create_dict_from_split_index(self, split_index, value):
        return reduce(lambda res, cur: {cur: res}, reversed(split_index), value)

    def _update_dict(self, d, u):
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = self._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _build_dtype(self, component, component_type):
        try:
            index_set_names = component.index_set().subsets(
                expand_all_set_operators=False
            )
            index_names = [set_.name for set_ in index_set_names]
            combined_header = ", ".join(index_names)
        except AttributeError:
            combined_header = "index"

        indices = (
            list(component.keys())
            if hasattr(component, "keys")
            else list(range(len(component)))
        )
        max_index_length = max((len(str(idx)) for idx in indices), default=1)

        dtype = [(combined_header, f"S{max_index_length + 10}")]

        if component_type == pyo.Var:
            dtype.append(("value", "f8"))
        elif component_type == pyo.Expression:
            max_expr_length = max(
                (len(str(component[idx].expr)) for idx in indices), default=100
            )
            dtype.extend([("expression", f"S{max_expr_length + 10}"), ("value", "f8")])
        elif component_type == pyo.Param:
            dtype.extend([("value", "f8"), ("unit", f"S50"), ("description", f"S100")])

        return dtype, combined_header

    def _fill_structured_array(self, array, component, component_type, combined_header):
        indices = list(component.keys())

        for i, index in enumerate(indices):
            index_str = (
                str(index)
                if not isinstance(index, tuple)
                else f"({', '.join(map(str, index))})"
            )
            array[combined_header][i] = index_str.encode("ascii", "ignore")

            if component_type == pyo.Var:
                try:
                    array["value"][i] = (
                        pyo.value(component[index])
                        if component[index].value is not None
                        else np.nan
                    )
                except Exception:
                    array["value"][i] = np.nan

            elif component_type == pyo.Expression:
                expr = component[index].expr
                array["expression"][i] = str(expr).encode("ascii", "ignore")
                try:
                    array["value"][i] = pyo.value(expr)
                except Exception:
                    array["value"][i] = np.nan

            elif component_type == pyo.Param:
                try:
                    array["value"][i] = pyo.value(component[index])
                except Exception:
                    array["value"][i] = np.nan

                doc = getattr(component, "doc", "")
                unit = extract_unit(doc) if doc else ""
                array["unit"][i] = safe_encode(unit)
                array["description"][i] = safe_encode(doc)
