
import json
import re
from collections.abc import Mapping
from functools import reduce
import h5py
import numpy as np
import pyomo.environ as pyo
from ruamel.yaml.scalarfloat import ScalarFloat


# Recursively save dictionary contents to an HDF5 group
def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively saves the contents of a dictionary to an HDF5 file group, creating datasets and groups as needed.

    Args:
        h5file (h5py.File): The HDF5 file where data will be saved.
        path (str): The current path in the HDF5 structure.
        dic (dict): The dictionary containing data to be saved.

    Raises:
        ValueError: If an unsupported data type is encountered.
    """
    for key, item in dic.items():
        try:
            if isinstance(item, dict):
                if "value" in item and "unit" in item:
                    dataset = h5file.create_dataset(path + str(key), data=item["value"])
                    dataset.attrs["unit"] = item["unit"]
                    if "description" in item:
                        dataset.attrs["description"] = item["description"]
                elif "value" in item and "description" in item:
                    group = h5file.create_group(path + str(key))
                    for subkey, subitem in item["value"].items():
                        if isinstance(subitem, (int, float, np.number)):
                            group.create_dataset(subkey, data=subitem)
                        elif isinstance(subitem, str):
                            group.create_dataset(subkey, data=np.string_(subitem))
                        elif isinstance(subitem, dict):
                            json_data = json.dumps(subitem)
                            group.create_dataset(subkey, data=np.string_(json_data))
                        else:
                            raise ValueError(f"Unsupported data type for subkey {subkey}: {type(subitem)}")
                    group.attrs["description"] = item["description"]
                else:
                    recursively_save_dict_contents_to_group(h5file, path + str(key) + "/", item)
            else:
                if isinstance(item, (int, float, np.number)):
                    h5file.create_dataset(path + str(key), data=item)
                elif isinstance(item, str):
                    h5file.create_dataset(path + str(key), data=np.string_(item))
                elif isinstance(item, list):
                    h5file.create_dataset(path + str(key), data=np.array(item, dtype=np.float64))
                elif item is None:
                    h5file.create_dataset(path + str(key), data=np.string_("None"))
                else:
                    raise ValueError(f"Unsupported data type for key {key}: {type(item)}")
        except Exception as e:
            print(f"Error saving {path}{key}: {e}")

# Save dictionary data to HDF5
def save_dict_to_hdf5(dic, h5file):
    """
    Saves the entire dictionary to an HDF5 file by calling the recursive function.

    Args:
        dic (dict): The dictionary to be saved.
        h5file (h5py.File): The HDF5 file object where the dictionary will be saved.
    """
    recursively_save_dict_contents_to_group(h5file, "/", dic)

# Convert ScalarFloat types to float
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
        return {key: convert_scalarfloats_to_floats(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_scalarfloats_to_floats(element) for element in obj]
    else:
        return obj

# Assign default units based on variable name
def assign_default_unit(var_name):
    """
    Assigns default units based on specific keywords in a variable name.

    Args:
        var_name (str): The name of the variable.

    Returns:
        str: The assigned unit (e.g., "Pa" for pressure), or an empty string if no match is found.
    """
    if "pressure" in var_name.lower():
        return "Pa"
    elif "volume" in var_name.lower():
        return "m³/s"
    elif "power" in var_name.lower():
        return "W"
    else:
        return ""


def save_components_as_structured_array(instance, h5file, component_type, group_name):
    """
    Saves Pyomo variables or expressions as structured arrays in an HDF5 file, with split columns for index names.

    Args:
        instance (pyomo.environ.ConcreteModel): The Pyomo model instance containing variables or expressions.
        h5file (h5py.File): The HDF5 file object where the data will be saved.
        component_type (type): The type of Pyomo component (pyo.Var or pyo.Expression).
        group_name (str): The group name under which the data will be saved.
    """
    # Determine the main group name based on the component type
    main_group = 'Variable' if component_type == pyo.Var else 'Expression'
    component_group = h5file.require_group(main_group)

    # Iterate over all active components of the given type
    for component in instance.component_objects(component_type, active=True):
        comp_name = str(component.name)
        match = re.match(r'scenario\[(\d+)\]\.(.+)', comp_name)
        if match:
            scenario_number = match.group(1)
            base_comp_name = match.group(2)
        else:
            scenario_number = None
            base_comp_name = comp_name

        # Set unit for variables, or description for both variables and expressions
        unit = assign_default_unit(base_comp_name) if component_type == pyo.Var else None
        description = getattr(component, "doc", "")

        # Prepare indices and index names
        indices = list(component.keys())
        index_set_names = component.index_set().subsets()
        index_names = [set_.name for set_ in index_set_names]

        # Define the dtype for the structured array based on number of index names
        if len(index_names) > 1:
            first_header, second_header = index_names[0], index_names[1]
            dtype = [(first_header, 'S100'), (second_header, 'S500')]
        else:
            first_header = index_names[0]
            dtype = [(first_header, 'S500')]

        # Add value or expression columns
        if component_type == pyo.Var:
            dtype.append(('value', 'f8'))
        elif component_type == pyo.Expression:
            dtype.append(("expression", "S1000"))  # String for expression
            dtype.append(("value", 'f8'))  # Float for evaluated value

        # Initialize the structured array
        structured_array = np.zeros(len(indices), dtype=dtype)

        # Populate the structured array
        for i, index in enumerate(indices):
            # Split index values based on the number of headers
            if isinstance(index, tuple):
                # Place the first two values in the first header column
                structured_array[first_header][i] = f"({', '.join(map(str, index[:2]))})".encode('utf-8')
                # Place the remaining values in the second header column, if available
                if len(index) > 2:
                    structured_array[second_header][i] = f"({', '.join(map(str, index[2:]))})".encode('utf-8')
            else:
                # If index is not a tuple, place it in the first header column
                structured_array[first_header][i] = str(index).encode('utf-8')
            
            # Add component-specific data
            if component_type == pyo.Var:
                structured_array['value'][i] = pyo.value(component[index])
            elif component_type == pyo.Expression:
                structured_array["expression"][i] = str(component[index].expr).encode('utf-8')
                structured_array["value"][i] = pyo.value(component[index].expr)

        # Organize data by scenario if applicable
        if scenario_number:
            scenario_group = component_group.require_group('Scenario')
            scenario_specific_group = scenario_group.require_group(scenario_number)

            if base_comp_name in scenario_specific_group:
                del scenario_specific_group[base_comp_name]
            dataset = scenario_specific_group.create_dataset(base_comp_name, data=structured_array)
            if component_type == pyo.Var:
                dataset.attrs['unit'] = str(unit) if unit else "No unit"
            dataset.attrs['description'] = str(description) if description else "No description"
        else:
            if base_comp_name in component_group:
                del component_group[base_comp_name]
            dataset = component_group.create_dataset(base_comp_name, data=structured_array)
            if component_type == pyo.Var:
                dataset.attrs['unit'] = str(unit) if unit else "No unit"
            dataset.attrs['description'] = str(description) if description else "No description"



# Main function to save all components to HDF5
def save_h5(instance, results, filepath):
    """
    Saves all components of a Pyomo model instance and solver results to an HDF5 file.

    Args:
        instance (pyomo.environ.ConcreteModel): The Pyomo model instance.
        results (dict): The results dictionary containing the solver output.
        filepath (str): The file path where the HDF5 file will be saved.
    """
    res_dict = create_results_dict(instance, results)
    res_dict = convert_scalarfloats_to_floats(res_dict)
    res_dict = replace_none_with_string(res_dict)

    with h5py.File(filepath, "w") as h5file:
        save_dict_to_hdf5(res_dict, h5file)
        save_components_as_structured_array(instance, h5file, pyo.Var, 'Variable')
        save_components_as_structured_array(instance, h5file, pyo.Expression, 'Expression')

# Utility functions for dictionary manipulation and results creation
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
    Splits a variable name to separate indices (handles cases with periods and square brackets).

    Args:
        name (str): The name of the variable.

    Returns:
        list: A list of split name components.
    """
    name = re.split(r"\[|\]|(?<!\d)[\[\].](?!\d)", name)
    return [x for x in name if x]

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

def create_results_dict(instance, results):
    """
    Creates a dictionary of model results, including sets, parameters, objectives, and constraints.

    Args:
        instance (pyomo.environ.ConcreteModel): The Pyomo model instance.
        results (dict): The results dictionary containing the solver output.

    Returns:
        dict: A dictionary containing the results organized by category.
    """
    def extract_data(component, category, res_dict):
        if category == "Variable":
            return
        for item in component:
            keys = split_index_name(str(item.name))
            try:
                value = pyo.value(item)
                doc = getattr(item, "doc", "")
                unit = extract_unit(doc) if doc else ""
                value_dict = {"value": value, "unit": unit, "description": doc if doc else ""}
            except ValueError:
                value_dict = {"value": "Variable not defined", "unit": "", "description": ""}
            res_dict[category] = update(res_dict.get(category, {}), create_dict_from_split_index(keys, value_dict))

    def extract_constraints(component, res_dict):
        for con in component:
            keys = split_index_name(str(con.name))
            constraint_info = {
                "body": str(con.body),
                "lower_bound": pyo.value(con.lower) if con.has_lb() else None,
                "upper_bound": pyo.value(con.upper) if con.has_ub() else None,
            }
            constraint_info = replace_none_with_string(constraint_info)
            doc = getattr(con, "doc", "")
            value_dict = {"value": constraint_info, "description": doc}
            res_dict["Constraint"] = update(res_dict.get("Constraint", {}), create_dict_from_split_index(keys, value_dict))

    res_dict = {
        "Set": {},
        "Parameter": {},
        "Objective": {},
        "Constraint": {},
        "Expression": {},
    }

    for s in instance.component_data_objects(pyo.Set, active=True):
        keys = split_index_name(s.name)
        doc = getattr(s, "doc", "")
        value = list(s.data())
        res_dict["Set"] = update(res_dict["Set"], create_dict_from_split_index(keys, {"value": value, "unit": "", "description": doc}))

    for p in instance.component_objects(pyo.Param, active=True):
        for index in p:
            keys = split_index_name(str(p.name))
            if index:
                keys.append(index)
            doc = getattr(p, "doc", "")
            value = p.extract_values()[index] if index else pyo.value(p)
            unit = extract_unit(doc) if doc else ""
            res_dict["Parameter"] = update(res_dict.get("Parameter", {}), create_dict_from_split_index(keys, {"value": value, "unit": unit, "description": doc}))

    extract_data(instance.component_data_objects(pyo.Objective, active=True), "Objective", res_dict)
    extract_constraints(instance.component_data_objects(pyo.Constraint, active=True), res_dict)

    res_dict["Problem Definition"] = {k: v.get_value() for k, v in results["Problem"].items()}
    res_dict["Solver Output"] = {k: v.get_value() for k, v in results["Solver"][0].items() if k != "Statistics"}

    return res_dict
