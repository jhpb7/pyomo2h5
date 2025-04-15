import h5py
import os
import subprocess
import numpy as np
import cloudpickle
from .instance_saver import InstanceSaver
from .dict_saver import DictSaver
from .log_saver import LogSaver
from .utils import convert_scalarfloats, replace_none
import pyomo.environ as pyo


class PyomoHDF5Saver(InstanceSaver, DictSaver, LogSaver):
    def __init__(self, filepath, mode="a", force=False):
        self.filepath = filepath if filepath.endswith(".h5") else filepath + ".h5"
        if os.path.exists(self.filepath) and not force and mode in ("w", "w-", "x"):
            raise FileExistsError(f"File '{self.filepath}' already exists.")
        self.file = h5py.File(self.filepath, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def solve_and_save_instance(
        self,
        instance,
        solver,
        solver_tee=True,
        save_log_flag=True,
        save_constraint_flag=True,
        pickle_flag=True,
        git_flag=True,
    ):
        solver.options["LogFile"] = self.filepath.replace("h5", "log")
        results = solver.solve(instance, tee=solver_tee)
        self.save_instance(
            instance,
            results,
            solver.options,
            save_log_flag,
            save_constraint_flag,
            pickle_flag,
            git_flag,
        )

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
        self._save_solver_results(results, solver_options)

        opt_group = self.file["Optimisation Components"]
        self._save_model_sets(instance, group=opt_group)
        self._save_objective(instance, group=opt_group)
        self._save_components(instance, pyo.Var, "Variable", group=opt_group)
        self._save_components(instance, pyo.Expression, "Expression", group=opt_group)
        self._save_components(instance, pyo.Param, "Parameter", group=opt_group)

        if save_constraint_flag:
            self._save_constraints(instance, group=opt_group)

        if pickle_flag:
            self._save_pickled_instance(instance)

        if save_log_flag:
            self._save_solver_log()

        if git_flag:
            self._save_git_hash()

    def save_tracked_constraints(self, tracker, group_name="TrackedConstraints"):
        group = self.file.require_group(group_name)

        try:
            self._serialize_constraints_to_dataset(
                tracker.constraint_log,
                group,
                dataset_name="tracked_constraints",
                description="Constraints tracked during runtime",
            )
        except Exception as e:
            print(f"Failed to save tracked constraints: {e}")

    def load_instance(self):
        pickled_data = self.file["pickled_pyomo_instance"][()]
        return cloudpickle.loads(pickled_data)

    def _save_solver_results(self, results, solver_options):
        res_dict = {
            "Optimisation Components": {
                "Set": {},
                "Parameter": {},
                "Objective": {},
                "Constraint": {},
                "Expression": {},
            },
            "Problem Definition": {
                k: v.get_value() for k, v in results["Problem"].items()
            },
            "Solver": {
                k: v.get_value()
                for k, v in results["Solver"][0].items()
                if k != "Statistics"
            },
        }

        if solver_options:
            res_dict["Solver"]["Options"] = dict(solver_options)

        res_dict = convert_scalarfloats(res_dict)
        res_dict = replace_none(res_dict)
        self._save_structured_dict(self.file, res_dict, "/")

    def _save_pickled_instance(self, instance):
        self.file.create_dataset(
            "pickled_pyomo_instance", data=np.void(cloudpickle.dumps(instance))
        )

    def _save_git_hash(self):
        try:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("utf-8")
                .strip()
            )
            self.file.create_dataset("Git Hash", data=git_hash)
        except Exception:
            pass  # Git might not be installed or we're not in a repo
