import h5py
import os
import subprocess
import cloudpickle
from .instance_saver import InstanceSaver
from .dict_saver import DictSaver
from .utils import convert_scalarfloats, replace_none


class PyomoHDF5Saver(InstanceSaver, DictSaver):
    def __init__(self, filepath, mode="w", force=False):
        self.filepath = filepath if filepath.endswith(".h5") else filepath + ".h5"
        if os.path.exists(self.filepath) and not force and mode in ("w", "w-", "x"):
            raise FileExistsError(f"File '{self.filepath}' already exists.")
        self.file = h5py.File(self.filepath, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def save_instance(
        self, instance, results=None, solver_options=None, save_constraint_flag=True
    ):
        res_dict = {"Solver": {}, "Parameter": {}, "Set": {}}  # Simplified
        res_dict = convert_scalarfloats(res_dict)
        res_dict = replace_none(res_dict)
        self._save_dict_with_metadata(self.file, res_dict, "/")
        if save_constraint_flag:
            self._save_constraints(instance, self.file)

    def save_tracked_constraints(self, tracker, group_name="TrackedConstraints"):
        group = self.file.require_group(group_name)
        for name, expr in tracker.constraint_log.items():
            group.create_dataset(name, data=str(expr))

    def load_instance(self):
        pickled_data = self.file["pickled_pyomo_instance"][()]
        return cloudpickle.loads(pickled_data)
