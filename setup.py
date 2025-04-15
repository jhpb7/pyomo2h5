from setuptools import setup, find_packages

setup(
    name="pyomo2h5",  # New package name
    version="0.2",  # Version
    description="A utility for saving Pyomo models and results to HDF5 format.",  # Short description
    author="Julius Breuer",  # Your name
    author_email="julius.breuer@tu-darmstadt.de",  # Your email
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "pyomo>=6.4.4",
        "h5py>=3.8",
        "cloudpickle>=2.0.0",
        "ruamel.yaml>=0.17.21",
        "numpy>=1.21",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Minimum Python version
)
