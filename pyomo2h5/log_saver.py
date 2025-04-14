import os
import numpy as np
import h5py
import re


class LogSaver:
    def _save_solver_log(self):
        log_path = self.filepath.replace(".h5", ".log")
        if os.path.exists(log_path):
            with open(log_path, "r") as file:
                log_content = file.read()
            solver_group = self.file.require_group("Solver")
            solver_group.create_dataset("Log", data=log_content.encode("utf-8"))
            self._parse_log(log_path)
            os.remove(log_path)

    def _parse_log(self, log_path, group_name="Solver", dataset_name="Convergence"):
        with open(log_path, "r") as f:
            lines = f.readlines()

        if not lines:
            print("Log file is empty. Convergence is not saved.")

        if lines[1].startswith("Gurobi"):
            self._parse_gurobi_log(lines, group_name, dataset_name)
        elif lines[2].startswith("SCIP"):
            self._parse_scip_log(lines, group_name, dataset_name)
        else:
            print("Did not recognise solver type - Convergence is not saved.")

    def _parse_scip_log(self, lines, group_name="Solver", dataset_name="Convergence"):

        # Step 1: Find the start of the progress table
        start_index = None
        for i, line in enumerate(lines):
            if "time" in line and "primalbound" in line:
                start_index = i + 1  # skip header and separator
                break

        if start_index is None:
            print(
                "Progress table not found. This either indicates the model is infeasible, unbounded or was solved instantly"
            )
            return

        # Step 2: Parse lines using whitespace splitting
        data = []
        for line in lines[start_index + 1 :]:
            if not line.strip():  # stop if the line is completely empty
                break
            tokens = [token.strip() for token in line.strip().split("|")]
            # Find gap (%), then get the two values before and the time (with 's')
            if tokens[0] == "time":
                continue
            bestbd = float(tokens[-3])
            time_str = re.sub(r"[a-zA-Z]", "", tokens[0])
            time = float(time_str)
            incumbent = float(tokens[-4])
            if tokens[-2].endswith("%"):
                gap = float(tokens[-2].strip("%"))
            else:
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

    def _parse_gurobi_log(self, lines, group_name="Solver", dataset_name="Convergence"):

        # Step 1: Find the start of the progress table
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

        # Step 2: Parse lines using whitespace splitting
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
