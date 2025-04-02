# Pyomo_to_hdf5

A Python library for efficiently saving Pyomo models, results, and optimization components to HDF5 format. This library provides utilities to handle Pyomo variables, expressions, parameters, and constraints, ensuring they are saved in a structured and accessible manner.

Key Features:

1. Save Pyomo models and results to HDF5 format with hierarchical structure.
2.  Handles Pyomo components such as Sets, Parameters, Variables, Constraints, Expressions, Objective(s)
3. Recursively saves dictionaries, including metadata like units and descriptions.
4. Automatically handles uninitialized variables and ensures robust error handling.
5. Converts None values, ScalarFloat, and other edge cases for compatibility with HDF5.
6. Includes the option of adding metadata and comments to the hdf5 file
7. Adds Constraint Tracker which stores certain constraints and thus makes them easier available e.g. for parameter-sweeps like pareto-fronts



# Authors

Ashar Pasha

Julius Breuer



# Install  / uninstall

Install using
`pip install pyomo2h5`

To uninstall use
`pip uninstall pyomo2h5`

