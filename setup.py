from setuptools import setup, find_packages

setup(
    name="pyomo2h5",  # New package name
    version="0.2",  # Version
    description="A utility for saving Pyomo models and results to HDF5 format.",  # Short description
    author="Your Name",  # Your name
    author_email="your.email@example.com",  # Your email
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "h5py",       # HDF5 library
        "numpy",      # NumPy for scientific computing
        "pyomo",      # Pyomo library
        "ruamel.yaml" # YAML library for handling configurations
    ],
    entry_points={
        "console_scripts": [
            # Example command-line script: save-hdf5 runs the `save_h5` function
            "save-hdf5=pyomo2h5.save_hdf5:save_h5",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
)
