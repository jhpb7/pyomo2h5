pyomo_to_hdf5
A Python library for efficiently saving Pyomo models, results, and optimization components to HDF5 format. This library provides utilities to handle Pyomo variables, expressions, parameters, and constraints, ensuring they are saved in a structured and accessible manner.

Key Features
Save Pyomo models and results to HDF5 format with hierarchical structure.
Handles Pyomo components such as:
Variables (pyo.Var)
Expressions (pyo.Expression)
Parameters (pyo.Param)
Constraints (pyo.Constraint)
Recursively saves dictionaries, including metadata like units and descriptions.
Automatically handles uninitialized variables and ensures robust error handling.
Converts None values, ScalarFloat, and other edge cases for compatibility with HDF5.
Includes utilities to handle default units for common optimization terms.

Example Code: 

import pyomo.environ as pyo
import pyomo_to_hdf5 as hdf5_saver
import h5py

# Create a simple Pyomo model
model = pyo.ConcreteModel()
model.x = pyo.Var(bounds=(0, 10), initialize=5)
model.y = pyo.Var(bounds=(0, 10), initialize=3)
model.obj = pyo.Objective(expr=model.x**2 + model.y**2)
model.c1 = pyo.Constraint(expr=model.x + model.y <= 10)

# Solve the model
solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

# Save the model and results to an HDF5 file
with h5py.File("model_results.h5", "w") as h5file:
    hdf5_saver.save_h5(model, results, h5file.name)


Saving a Custom Dictionary
You can save any dictionary (with metadata like units and descriptions) to HDF5:


import h5py
from pyomo_to_hdf5.save_hdf5 import save_dict_to_hdf5

data = {
    "pressure": {"value": 101325, "unit": "Pa", "description": "Standard atmospheric pressure"},
    "flow_rate": {"value": 1.2, "unit": "m³/s", "description": "Volume flow rate"}
}

with h5py.File("custom_data.h5", "w") as h5file:
    save_dict_to_hdf5(data, h5file)
print("Data saved successfully!")


Accessing Saved Data
To access saved HDF5 files:

import h5py

with h5py.File("model_results.h5", "r") as h5file:
    print(list(h5file.keys()))  # Inspect top-level groups
    print(h5file["Variable/Scenario/1/x"][:])  # Access a specific dataset



Directory Structure

pyomo-to-hdf5/
├── README.md                    # Documentation
├── requirements.txt             # Dependencies
├── setup.py                     # Installation script
├── pyomo_to_hdf5/               # Package directory
│   ├── __init__.py              # Package initialization
│   ├── save_hdf5.py             # Core functions
└── test                         # Example tests and usage scripts


Authors:

Ashar Pasha
Julius Breuer



Contributing
We welcome contributions! If you'd like to contribute:

Fork the repository.
Create a feature branch: git checkout -b feature-name.
Commit your changes: git commit -m "Add new feature".
Push to the branch: git push origin feature-name.
Open a Pull Request.


Acknowledgments
This library was inspired by the need for structured and scalable storage of optimization results, leveraging Pyomo and HDF5.