import os
import numpy as np
import h5py


class LogSaver:
    def _save_solver_log(self):
        log_path = self.filepath.replace(".h5", ".log")
        if os.path.exists(log_path):
            with open(log_path, "r") as file:
                log_content = file.read()
            solver_group = self.file.require_group("Solver")
            solver_group.create_dataset("Log", data=log_content.encode("utf-8"))
            self._parse_gurobi_log(log_path)
            os.remove(log_path)

    def _parse_gurobi_log(
        self, log_path, group_name="Solver", dataset_name="Convergence"
    ):
        with open(log_path, "r") as f:
            lines = f.readlines()

        # Check if it's a Gurobi log
        if not lines or not lines[1].startswith("Gurobi"):
            print("Not a Gurobi log file.")
            return

        # Step 2: Find the start of the progress table
        start_index = None
        for i, line in enumerate(lines):
            if "Nodes" in line and "Objective Bounds" in line:
                start_index = i + 2  # skip header and separator
                break

        if start_index is None:
            print(
                "Progress table not found. This either indicates the model is infeasible, unbounded or was solved instantly"
            )
            return

        # Step 3: Parse lines using whitespace splitting
        data = []
        for line in lines[start_index + 1 :]:
            if not line.strip():  # stop if the line is completely empty
                break
            tokens = line.strip().split()
            # Find gap (%), then get the two values before and the time (with 's')
            if tokens[-1].endswith("s"):  # token.endswith("%") and idx >= 2 and
                bestbd = float(tokens[-4])
                time_str = tokens[-1]
                time = float(time_str.rstrip("s"))
                if tokens[-3].endswith("%"):
                    incumbent = float(tokens[-5])
                    gap = float(tokens[-3].strip("%"))
                else:
                    incumbent = np.inf
                    gap = np.inf
                data.append([incumbent, bestbd, gap, time])

        if not data:
            raise ValueError(
                "When trying to save the convergence log. No valid progress rows found."
            )

        # Write to HDF5
        group = self.file.require_group(group_name)
        if dataset_name in group:
            del group[dataset_name]
        dtype = np.dtype(
            [
                ("Primal Bound", np.float64),
                ("Dual Bound", np.float64),
                ("Gap in %", np.float64),
                ("Time in s", np.float64),
            ]
        )
        structured_data = np.array([tuple(row) for row in data], dtype=dtype)
        group.create_dataset(dataset_name, data=structured_data)
