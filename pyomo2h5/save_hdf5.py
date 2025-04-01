import os

import subprocess
import re
from collections.abc import Mapping
from functools import reduce
import h5py
import numpy as np
import pyomo.environ as pyo
from ruamel.yaml.scalarfloat import ScalarFloat
import cloudpickle


def convert_to_string_array(data):
    """
    Converts various types of string data to HDF5-compatible format.

    Args:
        data: Input data that might be string or array of strings

    Returns:
        numpy array with ASCII-encoded strings
    """
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


def recursively_save_dict_contents_to_group(h5file, path, dic):
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
                        convert_to_string_array(item["value"])
                        if isinstance(item["value"], (str, list, np.ndarray))
                        else item["value"]
                    )
                    dataset = h5file.create_dataset(path + str(key), data=value_data)
                    dataset.attrs["unit"] = str(item["unit"]).encode("ascii", "ignore")
                    if "description" in item:
                        dataset.attrs["description"] = str(item["description"]).encode(
                            "ascii", "ignore"
                        )
                elif "value" in item and "description" in item:
                    group = h5file.create_group(path + str(key))
                    for subkey, subitem in item["value"].items():
                        if isinstance(subitem, (int, float, np.number)):
                            group.create_dataset(subkey, data=subitem)
                        else:
                            # Convert string or array data
                            converted_data = convert_to_string_array(subitem)
                            group.create_dataset(subkey, data=converted_data)
                    group.attrs["description"] = str(item["description"]).encode(
                        "ascii", "ignore"
                    )
                else:
                    recursively_save_dict_contents_to_group(
                        h5file, path + str(key) + "/", item
                    )
            else:
                # Handle non-dict types
                if isinstance(item, (int, float, np.number)):
                    h5file.create_dataset(path + str(key), data=item)
                elif isinstance(item, (str, list, np.ndarray)):
                    converted_data = convert_to_string_array(item)
                    h5file.create_dataset(path + str(key), data=converted_data)
                else:
                    raise ValueError(
                        f"Unsupported data type for key {key}: {type(item)}"
                    )
        except Exception as e:
            print(f"Error saving {path}{key}: {e}")


def save_dict_to_hdf5(dic, h5file):
    """
    Saves the entire dictionary to an HDF5 file by calling the recursive function.

    Args:
        dic (dict): The dictionary to be saved.
        h5file (h5py.File): The HDF5 file object where the dictionary will be saved.
    """
    recursively_save_dict_contents_to_group(h5file, "/", dic)


def convert_scalarfloats_to_floats(obj):
    """
    Recursively converts ScalarFloat and np.float64 types in an object to regular Python floats.

    Args:
        obj: The object to be converted (could be a scalar, list, or dictionary).

    Returns:
        obj: The same object with all ScalarFloat and np.float64 types converted to float.
    """
    if isinstance(obj, ScalarFloat):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {
            key: convert_scalarfloats_to_floats(value) for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_scalarfloats_to_floats(element) for element in obj]
    else:
        return obj


def assign_default_unit(var_name):
    """
    Assigns default units based on specific keywords in a variable name.

    Args:
        var_name (str): The name of the variable.

    Returns:
        str: The assigned unit (e.g., "Pa" for pressure), or an empty string if no match is found.
    """
    # if "pressure" in var_name.lower():
    #     return "Pa"
    # elif "volume" in var_name.lower():
    #     return "m³/s"
    # elif "power" in var_name.lower():
    #     return "W"
    # else:
    return ""


def save_components_as_structured_array(instance, h5file, component_type, group_name):
    """
    Saves Pyomo variables, expressions, or parameters as structured arrays in an HDF5 file.
    Handles uninitialized variables, suppresses subset warnings, and ensures robustness.

    Args:
        instance (pyomo.environ.ConcreteModel): The Pyomo model instance.
        h5file (h5py.File): The HDF5 file object.
        component_type (type): The type of Pyomo component (e.g., pyo.Var, pyo.Expression, pyo.Param).
        group_name (str): The group name for saving (e.g., 'Variable', 'Expression', 'Parameter').
    """

    def safe_encode(value, default=""):
        """
        Safely encodes a value to ASCII, replacing None with a default string.
        """
        return str(value if value is not None else default).encode("ascii", "ignore")

    main_group = group_name
    component_group = h5file.require_group(main_group)

    for component in instance.component_objects(component_type, active=True):
        try:
            comp_name = str(component.name)

            # Enhanced regex to parse component names
            match = re.match(r"scenario\[(?:'([^']+)'|([^]]+))\]\.(.+)", comp_name)
            if match:
                scenario_number = match.group(1) or match.group(
                    2
                )  # Scenario number (e.g., '1')
                remaining_path = match.group(3)  # Everything after the scenario
                path_parts = re.split(r"\.|\[|\]", remaining_path)
                path_parts = [p for p in path_parts if p]  # Remove empty parts
            else:
                scenario_number = None
                path_parts = [comp_name]

            # Extract index names from the index_set
            try:
                index_set_names = component.index_set().subsets(
                    expand_all_set_operators=False
                )
                index_names = [set_.name for set_ in index_set_names]
                combined_header = ", ".join(index_names)
            except AttributeError:
                # Fallback for components without index_set
                combined_header = "index"

            # Get component indices
            indices = list(component.keys())
            max_index_length = max(len(str(idx)) for idx in indices) if indices else 1

            # Define dtype for structured array based on component type
            dtype = [(combined_header, f"S{max_index_length + 10}")]
            if component_type == pyo.Var:
                dtype.append(("value", "f8"))
            elif component_type == pyo.Expression:
                max_expr_length = (
                    max(len(str(component[idx].expr)) for idx in indices)
                    if indices
                    else 100
                )
                dtype.extend(
                    [("expression", f"S{max_expr_length + 10}"), ("value", "f8")]
                )
            elif component_type == pyo.Param:
                dtype.extend(
                    [("value", "f8"), ("unit", f"S50"), ("description", f"S100")]
                )

            # Create structured array
            structured_array = np.zeros(len(indices), dtype=dtype)
            for i, index_tuple in enumerate(indices):
                index_str = (
                    str(index_tuple)
                    if not isinstance(index_tuple, tuple)
                    else f"({', '.join(map(str, index_tuple))})"
                )
                structured_array[combined_header][i] = index_str.encode(
                    "ascii", "ignore"
                )
                # Handle variable, expression, or parameter values
                if component_type == pyo.Var:
                    try:
                        structured_array["value"][i] = (
                            pyo.value(component[index_tuple])
                            if component[index_tuple].value is not None
                            else np.nan
                        )
                    except ValueError:
                        # print("error saving")
                        structured_array["value"][i] = np.nan  # Graceful fallback
                elif component_type == pyo.Expression:
                    expr_str = str(component[index_tuple].expr)
                    structured_array["expression"][i] = expr_str.encode(
                        "ascii", "ignore"
                    )
                    # structured_array["description"][i] = safe_encode(doc)
                    try:
                        structured_array["value"][i] = pyo.value(
                            component[index_tuple].expr
                        )
                    except ValueError:
                        structured_array["value"][i] = np.nan
                elif component_type == pyo.Param:
                    try:
                        structured_array["value"][i] = pyo.value(component[index_tuple])
                    except ValueError:
                        structured_array["value"][i] = np.nan
                    # Handle unit and description attributes
                    doc = getattr(component, "doc", "")
                    unit = extract_unit(doc) if doc else ""
                    structured_array["unit"][i] = safe_encode(unit)
                    structured_array["description"][i] = safe_encode(doc)

            # Save structured array to HDF5
            if scenario_number:
                if "/Scenario" not in component_group:
                    scenario_group = component_group.require_group(
                        "Scenario"
                    )  # Top-level Scenario group
                current_group = scenario_group.require_group(scenario_number)
                for part in path_parts[
                    :-1
                ]:  # Exclude the last part (name or attribute)
                    current_group = current_group.require_group(part)

                dataset_name = path_parts[-1]
                if dataset_name in current_group:
                    del current_group[dataset_name]
                dataset = current_group.create_dataset(
                    dataset_name, data=structured_array
                )
            else:
                dataset_name = path_parts[0] if path_parts else "value"
                if dataset_name in component_group:
                    del component_group[dataset_name]
                dataset = component_group.create_dataset(
                    dataset_name, data=structured_array
                )

            # Add attributes to the dataset
            unit = (
                assign_default_unit(path_parts[-1])
                if component_type == pyo.Var
                else None
            )
            description = getattr(component, "doc", "")
            dataset.attrs["unit"] = safe_encode(unit)
            dataset.attrs["description"] = safe_encode(description)

        except Exception as e:
            print(f"Error saving component {comp_name}: {e}")

    # Print variables for inspection
    # for v in instance.component_objects(component_type, active=True):
    #     print(f"Component: {v.name}")
    #     for index in v:
    #         try:
    #             print(f"  {index} = {pyo.value(v[index])}")
    #         except ValueError:
    #             print(f"  {index} = Uninitialized")


def save_sets(instance, res_dict):
    # Extract sets
    for s in instance.component_data_objects(pyo.Set, active=True):
        keys = split_index_name(s.name)
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

        res_dict["Set"] = update(
            res_dict["Set"],
            create_dict_from_split_index(
                keys, {"value": value, "unit": "", "description": doc}
            ),
        )
    return res_dict


def save_components_as_structured_array_for_constraints(instance, h5file, group_name):
    """
    Saves Pyomo constraints as structured arrays in an HDF5 file, organized by scenario.

    Args:
        instance (pyomo.environ.ConcreteModel): The Pyomo model instance.
        h5file (h5py.File): The HDF5 file object.
        group_name (str): The group name for saving (e.g., 'Constraint').
    """
    main_group = group_name
    component_group = h5file.require_group(main_group)

    for component in instance.component_objects(pyo.Constraint, active=True):
        try:
            comp_name = str(component.name)

            # Enhanced regex to parse variable names
            match = re.match(r"scenario\[(?:'([^']+)'|([^]]+))\]\.(.+)", comp_name)
            if match:
                # Extract scenario and the remaining variable hierarchy
                scenario_number = match.group(1) or match.group(
                    2
                )  # Scenario number (e.g., '1')
                remaining_path = match.group(
                    3
                )  # Everything after the scenario (e.g., 'fan_station[0a,0FAN].fans[Fan4,2].power')
                path_parts = re.split(
                    r"\.|\[|\]", remaining_path
                )  # Split on '.', '[', or ']'
                path_parts = [p for p in path_parts if p]  # Remove empty parts
            else:
                # Handle constraints without a scenario
                scenario_number = None
                path_parts = [comp_name]

            # Get component indices
            indices = list(component.keys())

            # Create dtype for structured array
            dtype = [
                ("body", "S1000"),  # Symbolic representation of the body
                ("value", "f8"),  # Evaluated numeric value of the body
                ("lower_bound", "f8"),  # Lower bound
                ("upper_bound", "f8"),  # Upper bound
            ]

            # Create structured array
            structured_array = np.zeros(len(indices), dtype=dtype)
            for i, index_tuple in enumerate(indices):
                structured_array["body"][i] = str(component[index_tuple].body).encode(
                    "ascii", "ignore"
                )
                structured_array["value"][i] = pyo.value(component[index_tuple].body)

                # Handle bounds
                structured_array["lower_bound"][i] = (
                    pyo.value(component[index_tuple].lower)
                    if component[index_tuple].has_lb()
                    else np.nan
                )
                structured_array["upper_bound"][i] = (
                    pyo.value(component[index_tuple].upper)
                    if component[index_tuple].has_ub()
                    else np.nan
                )

            # Save structured array to HDF5
            if scenario_number:
                if "/Scenario" not in component_group:
                    scenario_group = component_group.require_group(
                        "Scenario"
                    )  # Top-level Scenario group
                # Start at the scenario group
                current_group = scenario_group.require_group(scenario_number)

                # Create hierarchical groups for path parts
                for part in path_parts[
                    :-1
                ]:  # Exclude the last part (attribute or variable name)
                    current_group = current_group.require_group(part)

                # Save at the deepest level
                dataset_name = path_parts[
                    -1
                ]  # The last part is the variable name or attribute
                if dataset_name in current_group:
                    del current_group[dataset_name]
                dataset = current_group.create_dataset(
                    dataset_name, data=structured_array
                )
            else:
                # Flat constraint without a scenario
                dataset_name = path_parts[0] if path_parts else "value"
                if dataset_name in component_group:
                    del component_group[dataset_name]
                dataset = component_group.create_dataset(
                    dataset_name, data=structured_array
                )

            # Add attributes to the dataset
            description = getattr(component, "doc", "")
            if description:
                dataset.attrs["description"] = str(description).encode(
                    "ascii", "ignore"
                )

        except Exception as e:
            print(f"Error saving constraint {comp_name}: {e}")


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def save_h5(
    instance,
    results,
    filepath,
    solver_options=None,
    save_log_flag=True,
    save_constraint_flag=True,
    pickle_flag=True,
    git_flag=True,
):
    """
    Saves all components of a Pyomo model instance and solver results to an HDF5 file.

    Args:
        instance (pyomo.environ.ConcreteModel): The Pyomo model instance.
        results (dict): The results dictionary containing the solver output.
        filepath (str): The file path where the HDF5 file will be saved.
    """

    # Initialize results dictionary
    res_dict = {
        "Set": {},
        "Parameter": {},
        "Objective": {},
        "Constraint": {},
        "Expression": {},
    }

    res_dict = create_results_dict(results, solver_options, res_dict)
    res_dict = convert_scalarfloats_to_floats(res_dict)
    res_dict = replace_none_with_string(res_dict)

    res_dict = save_sets(instance, res_dict)

    with h5py.File(filepath + ".h5", "w") as h5file:
        save_dict_to_hdf5(res_dict, h5file)
        save_components_as_structured_array(instance, h5file, pyo.Var, "Variable")
        save_components_as_structured_array(
            instance, h5file, pyo.Expression, "Expression"
        )
        save_components_as_structured_array(instance, h5file, pyo.Param, "Parameter")
        if save_constraint_flag:
            save_components_as_structured_array_for_constraints(
                instance, h5file, "Constraint"
            )

        if pickle_flag:
            pickled_data = cloudpickle.dumps(instance)
            pickled_np = np.void(pickled_data)
            h5file.create_dataset("pickled_pyomo_instance", data=pickled_np)
        if save_log_flag:
            with open(filepath + ".log", "r") as file:
                log_content = file.read()
            h5file.create_dataset("solver_log_file", data=log_content.encode("utf-8"))
            os.remove(filepath + ".log")
        if git_flag:
            git_hash = get_git_hash()
            h5file.create_dataset("Git Hash", data=git_hash)


def save_dict_w_metadata(filename, data_dict, path_in_h5="/"):
    with h5py.File(filename + ".h5", "a") as h5file:
        save_dict_w_metadata_to_hdf5(h5file, data_dict, path_in_h5)


def save_dict_w_metadata_to_hdf5(h5file, data_dict, path="/"):
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
            save_dict_w_metadata_to_hdf5(h5file, val, subpath)


def load_instance_from_h5(filepath):
    """
    Load pickled instance from h5 by supplying filepath
    """
    # Load the Pyomo model
    with h5py.File(filepath, "r") as hdf5_file:
        # Read the stored bytes
        pickled_data = hdf5_file["pickled_pyomo_instance"][()]  # Read as bytes

        # Deserialize using cloudpickle
        instance = cloudpickle.loads(pickled_data)

    return instance


def update(d, u):
    """
    Recursively updates a nested dictionary with values from another dictionary.

    Args:
        d (dict): The original dictionary.
        u (dict): The dictionary with updates to be merged into the original dictionary.

    Returns:
        dict: The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def split_index_name(name):
    """
    Splits a variable name to separate indices (handles cases with periods and square brackets),
    but does not split if the period is between digits (e.g., '3.14').

    Args:
        name (str): The name of the variable.

    Returns:
        list: A list of split name components.
    """
    # Split on: '[' or ']', or '.' not between digits
    parts = re.split(r"\[|\]|(?<!\d)\.(?!\d)", name)
    return [x for x in parts if x]


def create_dict_from_split_index(split_index, value):
    """
    Creates a nested dictionary structure based on split index names.

    Args:
        split_index (list): List of split index components.
        value: The value to be stored in the nested dictionary.

    Returns:
        dict: A nested dictionary with the provided value.
    """
    return reduce(lambda res, cur: {cur: res}, reversed(split_index), value)


def extract_unit(doc):
    """
    Extracts the unit from a docstring if it is present within parentheses.

    Args:
        doc (str): The docstring containing the unit.

    Returns:
        str: The extracted unit (or an empty string if no unit is found).
    """
    match = re.search(r"\(([^)]+)\)", doc)
    return match.group(1) if match else ""


def replace_none_with_string(obj):
    """
    Replaces None values with the string "None" to avoid issues when saving.

    Args:
        obj: The object (dict, list, or value) to process.

    Returns:
        obj: The same object with None values replaced by the string "None".
    """
    if obj is None:
        return "None"
    elif isinstance(obj, dict):
        return {key: replace_none_with_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_none_with_string(element) for element in obj]
    else:
        return obj


def create_results_dict(results, solver_options, res_dict):
    """
    Creates a dictionary of model pyomo's results

    Args:
        results (dict): The results dictionary containing the solver output.

    Returns:
        dict: A dictionary containing the results organized by category.
    """
    res_dict["Problem Definition"] = {
        k: v.get_value() for k, v in results["Problem"].items()
    }
    res_dict["Solver"] = {
        k: v.get_value() for k, v in results["Solver"][0].items() if k != "Statistics"
    }
    if solver_options:
        res_dict["Solver"]["Options"] = dict(solver_options)

    return res_dict
