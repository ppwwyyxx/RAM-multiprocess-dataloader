from __future__ import annotations
from collections import defaultdict
import pickle
import sys
import torch
import json
from typing import Any
from tabulate import tabulate
import os
import time
import psutil


def get_mem_info(pid: int) -> dict[str, int]:
  res = defaultdict(int)
  for mmap in psutil.Process(pid).memory_maps():
    res['rss'] += mmap.rss
    res['pss'] += mmap.pss
    res['uss'] += mmap.private_clean + mmap.private_dirty
    res['shared'] += mmap.shared_clean + mmap.shared_dirty
    if mmap.path.startswith('/'):
      res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
  return res


class MemoryMonitor():
  def __init__(self, pids: list[int] = None):
    if pids is None:
      pids = [os.getpid()]
    self.pids = pids

  def add_pid(self, pid: int):
    assert pid not in self.pids
    self.pids.append(pid)

  def _refresh(self):
    self.data = {pid: get_mem_info(pid) for pid in self.pids}
    return self.data

  def table(self) -> str:
    self._refresh()
    table = []
    keys = list(list(self.data.values())[0].keys())
    now = str(int(time.perf_counter() % 1e5))
    for pid, data in self.data.items():
      table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
    return tabulate(table, headers=["time", "PID"] + keys)

  def str(self):
    self._refresh()
    keys = list(list(self.data.values())[0].keys())
    res = []
    for pid in self.pids:
      s = f"PID={pid}"
      for k in keys:
        v = self.format(self.data[pid][k])
        s += f", {k}={v}"
      res.append(s)
    return "\n".join(res)

  @staticmethod
  def format(size: int) -> str:
    for unit in ('', 'K', 'M', 'G'):
      if size < 1024:
        break
      size /= 1024.0
    return "%.1f%s" % (size, unit)


def create_coco() -> list[Any]:
  # Download from https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_train2017.json
  with open("instances_train2017.json") as f:
    obj = json.load(f)
    return obj["annotations"]


def read_sample(x):
  # A function that is supposed to read object x, incrementing its refcount.
  # This mimics what a real dataloader would do.
  if sys.version_info >= (3, 10, 6):
    # Before this version, pickle does not increment refcount. This is a bug that's
    # fixed in https://github.com/python/cpython/pull/92931.
    return pickle.dumps(x)
  else:
    import msgpack
    return msgpack.dumps(x)


class DatasetFromList(torch.utils.data.Dataset):
  def __init__(self, lst):
    self.lst = lst
  def __len__(self):
    return len(self.lst)
  def __getitem__(self, idx: int):
    return self.lst[idx]


if __name__ == "__main__":
  from serialize import NumpySerializedList
  monitor = MemoryMonitor()
  print("Initial", monitor.str())
  lst = create_coco()
  print("JSON", monitor.str())
  lst = NumpySerializedList(lst)
  print("Serialized", monitor.str())
  del lst; import gc; gc.collect()
  print("End", monitor.str())
