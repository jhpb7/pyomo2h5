import pyomo.environ as pyo


class ConstraintTracker:
    def __init__(self):
        self.constraint_log = {}
        self.counter = 0

    def add(self, constraint_obj, name=None):
        cname = name or constraint_obj.name
        if cname in self.constraint_log:
            raise ValueError(f"Constraint '{cname}' already exists.")
        self.constraint_log[cname] = constraint_obj
        self.counter += 1

    def delete(self, constraint_obj):
        if constraint_obj in self.constraint_log.values():
            self.constraint_log = {
                key: val
                for key, val in self.constraint_log.items()
                if val != constraint_obj
            }

    def print_constraints(self):
        for name, expr in self.constraint_log.items():
            print(f"{name}: {expr}")
