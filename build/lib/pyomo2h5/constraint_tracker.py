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

    def delete(self, model, name):
        if name in self.constraint_log and hasattr(model, name):
            model.del_component(getattr(model, name))
            del self.constraint_log[name]

    def print_constraints(self):
        for name, expr in self.constraint_log.items():
            print(f"{name}: {expr}")
