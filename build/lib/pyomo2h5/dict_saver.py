import h5py
import numpy as np
from .utils import safe_encode


class DictSaver:
    def _save_dict_with_metadata(
        self, h5file: h5py.File, data_dict: dict, path: str
    ) -> None:
        metadata = data_dict.get("Metadata", {})
        grp = h5file.require_group(path)
        for k, v in metadata.items():
            grp.attrs[k] = v

        for key, val in data_dict.items():
            if key == "Metadata":
                continue
            subpath = f"{path}/{key}"
            if isinstance(val, dict) and "Content" in val:
                dset = grp.create_dataset(key, data=str(val["Content"]))
                for mk, mv in val.get("Metadata", {}).items():
                    dset.attrs[mk] = mv
            elif isinstance(val, dict):
                self._save_dict_with_metadata(h5file, val, subpath)
