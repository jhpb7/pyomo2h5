# Pyomo2h5

A Python library for efficiently saving **Pyomo models, results, and optimization components** to **HDF5 format**.  
It provides utilities to handle Pyomo variables, expressions, parameters, and constraints, ensuring they are saved in a structured and accessible manner.

---

## ✨ Features

- Save Pyomo models and results to HDF5 format with hierarchical structure  
- Handle Pyomo components such as **Sets, Parameters, Variables, Constraints, Expressions, Objective(s)**  
- Recursively save dictionaries, including metadata like units and descriptions  
- Automatically handle uninitialized variables and ensure robust error handling  
- Convert `None`, `ScalarFloat`, and other edge cases for compatibility with HDF5  
- Add metadata and comments to the HDF5 file  
- Built-in **Constraint Tracker** to store selected constraints for tasks like parameter sweeps (e.g. Pareto fronts)  

---

## ⚡ Installation

Install locally:
```bash
pip install .
```
If you want to directly install from GitHub (without cloning):
```bash
pip install git+https://github.com/<your-username>/pyomo2h5.git
```
## Examples
📓 Try it out: [Example Notebook](./test.ipynb)

## Authors

- Julius Breuer
- Ashar Pasha

## Contributing
Contributions are welcome!
Open an issue for bug reports, ideas, or feature requests.
Submit a pull request with improvements.
If you plan a larger contribution, feel free to start a discussion first.

## 📜 License

This project is licensed under the terms of the MIT License

## Install  / uninstall

Install using
`pip install pyomo2h5`

To uninstall use
`pip uninstall pyomo2h5`

## Contributing
Contributions are welcome!  
Please open an issue or submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.


## Funding
The presented results were obtained within the research project ‘‘Algorithmic System Planning of Air Handling Units’’, Project No. 22289 N/1, funded by the program for promoting the Industrial Collective Research (IGF) of the German Ministry of Economic Affairs and Climate Action (BMWK), approved by the Deutsches Zentrum für Luft- und Raumfahrt (DLR). We want to thank all the participants of the working group for the constructive collaboration.
