import h5py
import numpy as np
from .utils import safe_encode
import json


class DictSaver:
    def _save_structured_dict(
        self, h5file: h5py.File, data_dict: dict, path: str = "/"
    ) -> None:
        """
        Recursively saves a regular dict structure into HDF5.

        For example:
            {
                "Solver": {
                    "Status": "ok",
                    "Time": 1.23
                },
                "Problem": {
                    "Upper bound": 0.0,
                    ...
                }
            }
        """
        group = h5file.require_group(path)

        for key, val in data_dict.items():
            subpath = f"{path}/{key}"

            if isinstance(val, dict):
                self._save_structured_dict(h5file, val, subpath)

            elif isinstance(val, (int, float, str, np.number)):
                group.create_dataset(key, data=val)

            elif isinstance(val, (list, np.ndarray)):
                group.create_dataset(key, data=self._convert_to_string_array(val))

            else:
                print(
                    f"[Warning] Skipping unsupported type at '{subpath}': {type(val)}"
                )

    def save_annotated_dict(
        self, data_dict: dict, path: str = "/", float_precision: int = 2
    ) -> None:
        """
        Recursively saves annotated content (with optional Metadata) into HDF5.

        Floats in datasets are rounded to `float_precision` decimals before saving.

        Parameters:
            data_dict (dict): Data to save.
            path (str): Root group path to start writing in HDF5.
            float_precision (int): Number of decimal places for float rounding.
        """

        h5file = self.file
        group = h5file.require_group(path)

        for key, val in data_dict.items():
            if key == "Metadata":
                for mk, mv in val.items():
                    group.attrs[mk] = mv
                continue

            subpath = f"{path}/{key}"

            if isinstance(val, dict) and "Content" in val:
                content = val["Content"]

                # Round floats if it's a float, list of floats, or ndarray
                if isinstance(content, float):
                    content = round(content, float_precision)
                elif isinstance(content, list):
                    content = [
                        round(x, float_precision) if isinstance(x, float) else x
                        for x in content
                    ]
                elif isinstance(content, np.ndarray) and np.issubdtype(
                    content.dtype, np.floating
                ):
                    content = np.round(content, float_precision)

                dset = group.create_dataset(key, data=content)

                for mk, mv in val.get("Metadata", {}).items():
                    dset.attrs[mk] = mv

            elif isinstance(val, dict):
                self.save_annotated_dict(val, subpath, float_precision=float_precision)

            else:
                print(f"[Warning] Unsupported format for key '{key}' — skipping")

    def _convert_to_string_array(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if data.dtype.kind in {"U", "O"}:
            max_len = max(len(str(item)) for item in data.flat)
            return np.array(
                [str(item).encode("ascii", "ignore") for item in data],
                dtype=f"S{max_len}",
            )
        return data
