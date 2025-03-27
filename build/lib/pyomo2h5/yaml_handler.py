from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarfloat import ScalarFloat
from collections import OrderedDict as ordereddict
import numpy as np


def convert_numpy_to_native(obj):
    """
    Recursively converts:
    - NumPy arrays to Python lists
    - NumPy float/int scalars to Python float/int
    - Preserves other data types
    """
    if isinstance(obj, np.ndarray):  # Convert NumPy array to list
        return obj.tolist()
    elif isinstance(
        obj, (np.float64, np.float32)
    ):  # Convert NumPy float to Python float
        return float(obj)
    elif isinstance(obj, range):
        return list(obj)
    elif isinstance(obj, (np.int64, np.int32)):  # Convert NumPy int to Python int
        return int(obj)
    elif isinstance(obj, dict):  # Recursively process dictionary values
        return {
            convert_numpy_to_native(key): convert_numpy_to_native(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):  # Recursively process list elements
        return [convert_numpy_to_native(element) for element in obj]
    elif isinstance(obj, tuple):  # Convert tuple elements recursively
        return tuple(convert_numpy_to_native(element) for element in obj)
    else:
        return obj  # Return as is if it's already a native type


def construct_yaml():

    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    # Custom constructor for floats
    def float_constructor(loader, node):
        value = loader.construct_scalar(node)
        return float(value)

    # Custom representer for tuples
    def tuple_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:python/tuple", data)

    yaml = YAML()
    yaml.constructor.add_constructor(
        "tag:yaml.org,2002:python/tuple", tuple_constructor
    )
    yaml.constructor.add_constructor("tag:yaml.org,2002:float", float_constructor)
    yaml.representer.add_representer(
        tuple, tuple_representer
    )  # Register the tuple representer
    return yaml


def convert_commentedmap_to_dict(obj):
    """
    Recursively converts ruamel.yaml-specific structures to built-in types.
    Handles CommentedMap, CommentedSeq, ScalarInt/Float for Pyomo compatibility.
    """
    if isinstance(obj, (CommentedMap, ordereddict, dict)):
        return {key: convert_commentedmap_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (CommentedSeq, list)):
        return [convert_commentedmap_to_dict(value) for value in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_commentedmap_to_dict(value) for value in obj)
    elif isinstance(obj, ScalarInt):
        return int(obj)
    elif isinstance(obj, ScalarFloat):
        return float(obj)
    else:
        return obj


def load_yaml(filename):
    """ "
    Safely load yaml files and yield them in correct format
    """
    yaml = construct_yaml()
    with open(filename, "r") as f:
        loadedfile = yaml.load(f)
    loadedfile = convert_commentedmap_to_dict(loadedfile)
    return loadedfile


def save_yaml(filename, data):
    """ "
    Safely dump data to yaml file. This code takes care of handling numpy-formated data structures etc
    """
    yaml = construct_yaml()
    with open(filename, "w") as f:
        yaml.dump(data, f)
