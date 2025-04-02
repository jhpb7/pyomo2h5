import h5py
import cloudpickle
import numpy as np
import pyomo.environ as pyo
from ruamel.yaml.scalarfloat import ScalarFloat
import re
import os
import subprocess
from collections.abc import Mapping
from functools import reduce


class PyomoHDF5Saver:
    def __init__(self, filepath, mode="a", force=False):
        self.filepath = filepath if filepath.endswith(".h5") else filepath + ".h5"

        if os.path.exists(self.filepath) and not force and mode in ("w", "w-", "x"):
            raise FileExistsError(
                f"File '{self.filepath}' already exists. Use force=True to overwrite."
            )

        if os.path.exists(self.filepath) and not force and mode == "a":
            # Still allow append, just warn (optional)
            pass

        self.file = h5py.File(self.filepath, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    # === Save methods ===
    def save_instance(
        self,
        instance,
        results,
        solver_options=None,
        save_log_flag=True,
        save_constraint_flag=True,
        pickle_flag=True,
        git_flag=True,
    ):

        res_dict = {
            "Set": {},
            "Parameter": {},
            "Objective": {},
            "Constraint": {},
            "Expression": {},
        }

        res_dict = self._create_results_dict(results, solver_options, res_dict)
        res_dict = self.convert_scalarfloats_to_floats(res_dict)
        res_dict = self.replace_none_with_string(res_dict)
        res_dict = self._save_sets(instance, res_dict)

        self._save_dict_to_hdf5(res_dict)
        self._save_components(instance, pyo.Var, "Variable")
        self._save_components(instance, pyo.Expression, "Expression")
        self._save_components(instance, pyo.Param, "Parameter")

        if save_constraint_flag:
            self._save_constraints(instance)

        if pickle_flag:
            self.file.create_dataset(
                "pickled_pyomo_instance", data=np.void(cloudpickle.dumps(instance))
            )

        if save_log_flag:
            with open(self.filepath.replace(".h5", ".log"), "r") as file:
                log_content = file.read()
            self.file.create_dataset(
                "solver_log_file", data=log_content.encode("utf-8")
            )
            os.remove(self.filepath.replace(".h5", ".log"))

        if git_flag:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("utf-8")
                .strip()
            )
            self.file.create_dataset("Git Hash", data=git_hash)

    def _save_dict_to_hdf5(self, dic):
        self._recursively_save_dict_contents_to_group("/", dic)

    def save_dict_with_metadata(self, data_dict, path="/"):
        self._save_dict_with_metadata(self.file, data_dict, path)

    # def _save_components(self, instance, component_type, group_name):
    #     def safe_encode(value, default=""):
    #         """
    #         Safely encodes a value to ASCII, replacing None with a default string.
    #         """
    #         return str(value if value is not None else default).encode(
    #             "ascii", "ignore"
    #         )

    #     main_group = group_name
    #     component_group = self.file.require_group(main_group)

    #     for component in instance.component_objects(component_type, active=True):
    #         try:
    #             comp_name = str(component.name)

    #             # Enhanced regex to parse component names
    #             match = re.match(r"scenario\[(?:'([^']+)'|([^]]+))\]\.(.+)", comp_name)
    #             if match:
    #                 scenario_number = match.group(1) or match.group(
    #                     2
    #                 )  # Scenario number (e.g., '1')
    #                 remaining_path = match.group(3)  # Everything after the scenario
    #                 path_parts = re.split(r"\.|\[|\]", remaining_path)
    #                 path_parts = [p for p in path_parts if p]  # Remove empty parts
    #             else:
    #                 scenario_number = None
    #                 path_parts = [comp_name]

    #             # Extract index names from the index_set
    #             try:
    #                 index_set_names = component.index_set().subsets(
    #                     expand_all_set_operators=False
    #                 )
    #                 index_names = [set_.name for set_ in index_set_names]
    #                 combined_header = ", ".join(index_names)
    #             except AttributeError:
    #                 # Fallback for components without index_set
    #                 combined_header = "index"

    #             # Get component indices
    #             indices = list(component.keys())
    #             max_index_length = (
    #                 max(len(str(idx)) for idx in indices) if indices else 1
    #             )

    #             # Define dtype for structured array based on component type
    #             dtype = [(combined_header, f"S{max_index_length + 10}")]
    #             if component_type == pyo.Var:
    #                 dtype.append(("value", "f8"))
    #             elif component_type == pyo.Expression:
    #                 max_expr_length = (
    #                     max(len(str(component[idx].expr)) for idx in indices)
    #                     if indices
    #                     else 100
    #                 )
    #                 dtype.extend(
    #                     [("expression", f"S{max_expr_length + 10}"), ("value", "f8")]
    #                 )
    #             elif component_type == pyo.Param:
    #                 dtype.extend(
    #                     [("value", "f8"), ("unit", f"S50"), ("description", f"S100")]
    #                 )

    #             # Create structured array
    #             structured_array = np.zeros(len(indices), dtype=dtype)
    #             for i, index_tuple in enumerate(indices):
    #                 index_str = (
    #                     str(index_tuple)
    #                     if not isinstance(index_tuple, tuple)
    #                     else f"({', '.join(map(str, index_tuple))})"
    #                 )
    #                 structured_array[combined_header][i] = index_str.encode(
    #                     "ascii", "ignore"
    #                 )
    #                 # Handle variable, expression, or parameter values
    #                 if component_type == pyo.Var:
    #                     try:
    #                         structured_array["value"][i] = (
    #                             pyo.value(component[index_tuple])
    #                             if component[index_tuple].value is not None
    #                             else np.nan
    #                         )
    #                     except ValueError:
    #                         # print("error saving")
    #                         structured_array["value"][i] = np.nan  # Graceful fallback
    #                 elif component_type == pyo.Expression:
    #                     expr_str = str(component[index_tuple].expr)
    #                     structured_array["expression"][i] = expr_str.encode(
    #                         "ascii", "ignore"
    #                     )
    #                     # structured_array["description"][i] = safe_encode(doc)
    #                     try:
    #                         structured_array["value"][i] = pyo.value(
    #                             component[index_tuple].expr
    #                         )
    #                     except ValueError:
    #                         structured_array["value"][i] = np.nan
    #                 elif component_type == pyo.Param:
    #                     try:
    #                         structured_array["value"][i] = pyo.value(
    #                             component[index_tuple]
    #                         )
    #                     except ValueError:
    #                         structured_array["value"][i] = np.nan
    #                     # Handle unit and description attributes
    #                     doc = getattr(component, "doc", "")
    #                     unit = PyomoHDF5Saver.extract_unit(doc) if doc else ""
    #                     structured_array["unit"][i] = safe_encode(unit)
    #                     structured_array["description"][i] = safe_encode(doc)

    #             # Save structured array to HDF5
    #             if scenario_number:
    #                 if "/Scenario" not in component_group:
    #                     scenario_group = component_group.require_group(
    #                         "Scenario"
    #                     )  # Top-level Scenario group
    #                 current_group = scenario_group.require_group(scenario_number)
    #                 for part in path_parts[
    #                     :-1
    #                 ]:  # Exclude the last part (name or attribute)
    #                     current_group = current_group.require_group(part)

    #                 dataset_name = path_parts[-1]
    #                 if dataset_name in current_group:
    #                     del current_group[dataset_name]
    #                 dataset = current_group.create_dataset(
    #                     dataset_name, data=structured_array
    #                 )
    #             else:
    #                 dataset_name = path_parts[0] if path_parts else "value"
    #                 if dataset_name in component_group:
    #                     del component_group[dataset_name]
    #                 dataset = component_group.create_dataset(
    #                     dataset_name, data=structured_array
    #                 )

    #             # Add attributes to the dataset
    #             unit = (
    #                 PyomoHDF5Saver.assign_default_unit(path_parts[-1])
    #                 if component_type == pyo.Var
    #                 else None
    #             )
    #             description = getattr(component, "doc", "")
    #             dataset.attrs["unit"] = safe_encode(unit)
    #             dataset.attrs["description"] = safe_encode(description)

    #         except Exception as e:
    #             print(f"Error saving component {comp_name}: {e}")

    # def save_constraints(self, instance):
    #     """
    #     Saves Pyomo constraints as structured arrays in an HDF5 file, organized by scenario.

    #     Args:
    #         instance (pyomo.environ.ConcreteModel): The Pyomo model instance.
    #         h5file (h5py.File): The HDF5 file object.
    #         group_name (str): The group name for saving (e.g., 'Constraint').
    #     """
    #     component_group = self.file.require_group("Constraint")

    #     for component in instance.component_objects(pyo.Constraint, active=True):
    #         try:
    #             comp_name = str(component.name)

    #             # Enhanced regex to parse variable names
    #             match = re.match(r"scenario\[(?:'([^']+)'|([^]]+))\]\.(.+)", comp_name)
    #             if match:
    #                 # Extract scenario and the remaining variable hierarchy
    #                 scenario_number = match.group(1) or match.group(
    #                     2
    #                 )  # Scenario number (e.g., '1')
    #                 remaining_path = match.group(
    #                     3
    #                 )  # Everything after the scenario (e.g., 'fan_station[0a,0FAN].fans[Fan4,2].power')
    #                 path_parts = re.split(
    #                     r"\.|\[|\]", remaining_path
    #                 )  # Split on '.', '[', or ']'
    #                 path_parts = [p for p in path_parts if p]  # Remove empty parts
    #             else:
    #                 # Handle constraints without a scenario
    #                 scenario_number = None
    #                 path_parts = [comp_name]

    #             # Get component indices
    #             indices = list(component.keys())

    #             # Create dtype for structured array
    #             dtype = [
    #                 ("body", "S1000"),  # Symbolic representation of the body
    #                 ("value", "f8"),  # Evaluated numeric value of the body
    #                 ("lower_bound", "f8"),  # Lower bound
    #                 ("upper_bound", "f8"),  # Upper bound
    #             ]

    #             # Create structured array
    #             structured_array = np.zeros(len(indices), dtype=dtype)
    #             for i, index_tuple in enumerate(indices):
    #                 structured_array["body"][i] = str(
    #                     component[index_tuple].body
    #                 ).encode("ascii", "ignore")
    #                 structured_array["value"][i] = pyo.value(
    #                     component[index_tuple].body
    #                 )

    #                 # Handle bounds
    #                 structured_array["lower_bound"][i] = (
    #                     pyo.value(component[index_tuple].lower)
    #                     if component[index_tuple].has_lb()
    #                     else np.nan
    #                 )
    #                 structured_array["upper_bound"][i] = (
    #                     pyo.value(component[index_tuple].upper)
    #                     if component[index_tuple].has_ub()
    #                     else np.nan
    #                 )

    #             # Save structured array to HDF5
    #             if scenario_number:
    #                 if "/Scenario" not in component_group:
    #                     scenario_group = component_group.require_group(
    #                         "Scenario"
    #                     )  # Top-level Scenario group
    #                 # Start at the scenario group
    #                 current_group = scenario_group.require_group(scenario_number)

    #                 # Create hierarchical groups for path parts
    #                 for part in path_parts[
    #                     :-1
    #                 ]:  # Exclude the last part (attribute or variable name)
    #                     current_group = current_group.require_group(part)

    #                 # Save at the deepest level
    #                 dataset_name = path_parts[
    #                     -1
    #                 ]  # The last part is the variable name or attribute
    #                 if dataset_name in current_group:
    #                     del current_group[dataset_name]
    #                 dataset = current_group.create_dataset(
    #                     dataset_name, data=structured_array
    #                 )
    #             else:
    #                 # Flat constraint without a scenario
    #                 dataset_name = path_parts[0] if path_parts else "value"
    #                 if dataset_name in component_group:
    #                     del component_group[dataset_name]
    #                 dataset = component_group.create_dataset(
    #                     dataset_name, data=structured_array
    #                 )

    #             # Add attributes to the dataset
    #             description = getattr(component, "doc", "")
    #             if description:
    #                 dataset.attrs["description"] = str(description).encode(
    #                     "ascii", "ignore"
    #                 )

    #         except Exception as e:
    #             print(f"Error saving constraint {comp_name}: {e}")

    def _save_components(self, instance, component_type, group_name):
        component_group = self.file.require_group(group_name)

        for component in instance.component_objects(component_type, active=True):
            try:
                comp_name = str(component.name)
                scenario, path_parts = self._parse_component_path(comp_name)

                dtype, combined_header = self._build_dtype(component, component_type)
                structured_array = np.zeros(len(component.keys()), dtype=dtype)
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
                self._add_metadata(dataset, component, component_type, path_parts)

            except Exception as e:
                print(f"Error saving component {comp_name}: {e}")

    def _save_constraints(self, instance):
        """
        Saves Pyomo constraints as structured arrays in an HDF5 file, organized by scenario.
        """
        component_group = self.file.require_group("Constraint")

        for component in instance.component_objects(pyo.Constraint, active=True):
            try:
                comp_name = str(component.name)
                scenario, path_parts = self._parse_component_path(comp_name)

                dtype = self._build_constraint_dtype()
                structured_array = np.zeros(len(set(component.keys())), dtype=dtype)
                self._fill_constraint_array(structured_array, component)

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

                description = getattr(component, "doc", "")
                if description:
                    dataset.attrs["description"] = self._safe_encode(description)

            except Exception as e:
                print(f"Error saving constraint {comp_name}: {e}")

    @staticmethod
    def _build_constraint_dtype():
        return [
            ("body", "S1000"),  # Symbolic representation of constraint body
            ("value", "f8"),  # Evaluated value
            ("lower_bound", "f8"),  # Lower bound
            ("upper_bound", "f8"),  # Upper bound
        ]

    @staticmethod
    def _fill_constraint_array(array, component):
        indices = list(component.keys())

        for i, index in enumerate(indices):
            con = component[index]
            array["body"][i] = str(con.body).encode("ascii", "ignore")

            try:
                array["value"][i] = pyo.value(con.body)
            except Exception:
                array["value"][i] = np.nan

            array["lower_bound"][i] = pyo.value(con.lower) if con.has_lb() else np.nan
            array["upper_bound"][i] = pyo.value(con.upper) if con.has_ub() else np.nan

    @staticmethod
    def _safe_encode(value, default=""):
        """
        Safely encode a string to ASCII, falling back to default if value is None.
        """
        return str(value if value is not None else default).encode("ascii", "ignore")

    @staticmethod
    def _parse_component_path(name):
        """
        Parses a Pyomo component name into scenario and path parts.
        """
        match = re.match(r"scenario\[(?:'([^']+)'|([^]]+))\]\.(.+)", name)
        if match:
            scenario = match.group(1) or match.group(2)
            remaining_path = match.group(3)
            path_parts = re.split(r"\.|\[|\]", remaining_path)
            path_parts = [p for p in path_parts if p]
        else:
            scenario = None
            path_parts = [name]
        return scenario, path_parts

    @staticmethod
    def _get_or_create_group(base_group, path_parts):
        """
        Walks or creates a nested group hierarchy.
        """
        current_group = base_group
        for part in path_parts:
            current_group = current_group.require_group(part)
        return current_group

    def _build_dtype(self, component, component_type):
        """
        Builds the dtype for a structured array based on component type.
        """
        try:
            index_set_names = component.index_set().subsets(
                expand_all_set_operators=False
            )
            index_names = [set_.name for set_ in index_set_names]
            combined_header = ", ".join(index_names)
        except AttributeError:
            combined_header = "index"

        indices = list(component.keys())
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

    def _add_metadata(self, dataset, component, component_type, path_parts):
        """
        Adds metadata to the HDF5 dataset.
        """
        if component_type == pyo.Var:
            unit = self.assign_default_unit(path_parts[-1])
        else:
            unit = getattr(component, "unit", "")

        description = getattr(component, "doc", "")

        dataset.attrs["unit"] = self._safe_encode(unit)
        dataset.attrs["description"] = self._safe_encode(description)

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
                unit = self.extract_unit(doc) if doc else ""
                array["unit"][i] = self._safe_encode(unit)
                array["description"][i] = self._safe_encode(doc)

    def _save_sets(self, instance, res_dict):
        for s in instance.component_data_objects(pyo.Set, active=True):
            keys = self.split_index_name(s.name)
            doc = getattr(s, "doc", "")
            try:
                if hasattr(s, "is_set_operator") and s.is_set_operator():
                    # For set operators like SetUnion, explicitly handle without descending
                    value = list(s) if hasattr(s, "__iter__") else []
                else:
                    # For regular sets, get the data directly
                    value = list(s.data())
            except AttributeError:
                value = list(s.data())  # Fallback for simple sets

            res_dict["Set"] = self.update(
                res_dict["Set"],
                self.create_dict_from_split_index(
                    keys, {"value": value, "unit": "", "description": doc}
                ),
            )
        return res_dict

    def load_instance(self):
        pickled_data = self.file["pickled_pyomo_instance"][()]
        return cloudpickle.loads(pickled_data)

    # === Tracking Methods ===

    def save_tracked_constraints(self, tracker, group_name="TrackedConstraints"):
        group = self.file.require_group(group_name)
        for name, expr in tracker.constraint_log.items():
            group.create_dataset(name, data=str(expr))

    # === Utility Methods ===
    def _recursively_save_dict_contents_to_group(self, path, dic):
        """
        Recursively saves the contents of a dictionary into an HDF5 group.

        Args:
            h5file (h5py.File): The HDF5 file object where the data will be saved.
            path (str): The current path in the HDF5 file.
            dic (dict): The dictionary to save into the HDF5 file.
        """
        for key, item in dic.items():
            try:
                if isinstance(item, dict):
                    if "value" in item and "unit" in item:
                        # Convert and save value
                        value_data = (
                            self.convert_to_string_array(item["value"])
                            if isinstance(item["value"], (str, list, np.ndarray))
                            else item["value"]
                        )
                        dataset = self.file.create_dataset(
                            path + str(key), data=value_data
                        )
                        dataset.attrs["unit"] = str(item["unit"]).encode(
                            "ascii", "ignore"
                        )
                        if "description" in item:
                            dataset.attrs["description"] = str(
                                item["description"]
                            ).encode("ascii", "ignore")
                    elif "value" in item and "description" in item:
                        group = self.file.create_group(path + str(key))
                        for subkey, subitem in item["value"].items():
                            if isinstance(subitem, (int, float, np.number)):
                                group.create_dataset(subkey, data=subitem)
                            else:
                                # Convert string or array data
                                converted_data = self.convert_to_string_array(subitem)
                                group.create_dataset(subkey, data=converted_data)
                        group.attrs["description"] = str(item["description"]).encode(
                            "ascii", "ignore"
                        )
                    else:
                        self._recursively_save_dict_contents_to_group(
                            path + str(key) + "/", item
                        )
                else:
                    # Handle non-dict types
                    if isinstance(item, (int, float, np.number)):
                        self.file.create_dataset(path + str(key), data=item)
                    elif isinstance(item, (str, list, np.ndarray)):
                        converted_data = self.convert_to_string_array(item)
                        self.file.create_dataset(path + str(key), data=converted_data)
                    else:
                        raise ValueError(
                            f"Unsupported data type for key {key}: {type(item)}"
                        )
            except Exception as e:
                print(f"Error saving {path}{key}: {e}")

    def _save_dict_with_metadata(self, h5file, data_dict, path):
        """
        Recursively saves a nested dictionary to an HDF5 file including metadata. The metadata can also be added for any other existing h5 group or dataset.

        The data_dict should be as follows:
        {"Metadata": {"Infos": "This will be metadata to the toplevel of the h5 file"}, "comment": {"Content": "This comes as data to the comment group", "Metadata": {"Info": "This will be the attribute Info and its text", "Written by": "Julius", "For": "Julius"}}}

        Args:
            h5file (h5py.File): An open HDF5 file object.
            path (str): Current path in the HDF5 file (e.g., "/constraints").
            data_dict (dict): Dictionary representing the group/data hierarchy.
                            Keys can be group names, dataset names, or 'metadata'.
        """

        metadata = data_dict.get("Metadata", {})

        # Save metadata to current path
        if path in h5file:
            grp = h5file[path]
        else:
            grp = h5file.create_group(path)
        for key, val in metadata.items():
            grp.attrs[key] = val

        for key, val in data_dict.items():
            if key == "Metadata":
                continue
            subpath = f"{path}/{key}"
            if isinstance(val, dict) and "Content" in val.keys():
                dset = grp.create_dataset(key, data=str(val["Content"]))
                if isinstance(val, dict) and "Metadata" in val:
                    for meta_key, meta_val in val["Metadata"].items():
                        dset.attrs[meta_key] = meta_val
            if isinstance(val, dict):
                self._save_dict_with_metadata(h5file, val, subpath)

    @staticmethod
    def convert_to_string_array(data):
        # Your existing logic
        if isinstance(data, (list, np.ndarray)):
            # Convert list or array of strings
            if isinstance(data, list):
                data = np.array(data)
            if data.dtype.kind in {"U", "O"}:  # Unicode or object dtype
                # Convert to ASCII strings with max length calculation
                max_length = max(len(str(item)) for item in data.flat)
                return np.array(
                    [str(item).encode("ascii", "ignore") for item in data],
                    dtype=f"S{max_length}",
                )
        elif isinstance(data, str):
            # Single string
            return np.bytes_(data.encode("ascii", "ignore"))
        return data

    @staticmethod
    def convert_scalarfloats_to_floats(obj):
        # Your logic here
        if isinstance(obj, ScalarFloat):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, dict):
            return {
                key: PyomoHDF5Saver.convert_scalarfloats_to_floats(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [
                PyomoHDF5Saver.convert_scalarfloats_to_floats(element)
                for element in obj
            ]
        else:
            return obj

    @staticmethod
    def replace_none_with_string(obj):
        # Your logic here
        if obj is None:
            return "None"
        elif isinstance(obj, dict):
            return {
                key: PyomoHDF5Saver.replace_none_with_string(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [PyomoHDF5Saver.replace_none_with_string(element) for element in obj]
        else:
            return obj

    def _create_results_dict(self, results, solver_options, res_dict):
        # Your logic here
        res_dict["Problem Definition"] = {
            k: v.get_value() for k, v in results["Problem"].items()
        }
        res_dict["Solver"] = {
            k: v.get_value()
            for k, v in results["Solver"][0].items()
            if k != "Statistics"
        }
        if solver_options:
            res_dict["Solver"]["Options"] = dict(solver_options)

        return res_dict

    @staticmethod
    def assign_default_unit(var_name):
        return ""

    @staticmethod
    def extract_unit(doc):
        match = re.search(r"\(([^)]+)\)", doc)
        return match.group(1) if match else ""

    @staticmethod
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = PyomoHDF5Saver.update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    @staticmethod
    def split_index_name(name):
        parts = re.split(r"\[|\]|(?<!\d)\.(?!\d)", name)
        return [x for x in parts if x]

    @staticmethod
    def create_dict_from_split_index(split_index, value):
        return reduce(lambda res, cur: {cur: res}, reversed(split_index), value)


class ConstraintTracker:
    def __init__(self):
        self.constraint_log = {}
        self.counter = 0

    def add(self, model, constraint_expr, name=None):
        """Add a constraint to the model and track it by name."""
        cname = name or f"added_constraint_{self.counter}"
        if cname in self.constraint_log:
            raise ValueError(f"Constraint '{cname}' already exists.")
        model.add_component(cname, pyo.Constraint(expr=constraint_expr))
        self.constraint_log[cname] = constraint_expr
        self.counter += 1

    def delete(self, model, name):
        """Remove a constraint by name from both model and tracker."""
        if name not in self.constraint_log:
            raise KeyError(f"Constraint '{name}' not found in tracker.")
        if hasattr(model, name):
            model.del_component(getattr(model, name))
        del self.constraint_log[name]

    def print_constraints(self):
        """Print all tracked constraints and their expressions."""
        if not self.constraint_log:
            print("No constraints tracked.")
        for name, expr in self.constraint_log.items():
            print(f"{name}: {expr}")
