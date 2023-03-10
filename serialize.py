# Copyright (c) Facebook, Inc. and its affiliates.
"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""

import multiprocessing as mp
import pickle
import numpy as np
import torch


class NumpySerializedList():
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)


# NOTE: https://github.com/facebookresearch/mobile-vision/pull/120
# has another implementation that does not use tensors.
class TorchShmSerializedList(TorchSerializedList):
    def __init__(self, lst: list):
        from detectron2.utils import comm

        if comm.get_local_rank() == 0:
            super().__init__(lst)
        if comm.get_local_size() == 1:
            # Just one GPU on this machine. Do nothing.
            return
        if comm.get_local_rank() == 0:
            # Move to shared memory, obtain a handle.
            serialized = bytes(mp.reduction.ForkingPickler.dumps(
                (self._addr, self._lst)))
            # Broadcast the handle of shared memory to other GPU workers.
            comm.all_gather(serialized)
        else:
            serialized = comm.all_gather(None)[comm.get_rank() - comm.get_local_rank()]
            # Materialize a tensor from shared memory.
            self._addr, self._lst = mp.reduction.ForkingPickler.loads(serialized)
            print(f"Worker {comm.get_rank()} obtains a dataset of length="
                  f"{len(self)} from its local leader.")
