#!/usr/bin/env python
import sys
import pickle
import time
import torch

from common import MemoryMonitor, create_list, DatasetFromList
from serialize import NumpySerializedList


def worker(_, dataset: torch.utils.data.Dataset):
  while True:
    for sample in dataset:
      # read the data, with a fake latency
      time.sleep(0.000001)
      result = pickle.dumps(sample)


if __name__ == "__main__":
  start_method = sys.argv[1] if len(sys.argv) == 2 else 'fork'

  monitor = MemoryMonitor()
  ds = DatasetFromList(NumpySerializedList(create_list()))
  print(monitor.table())

  ctx = torch.multiprocessing.start_processes(
      worker, (ds, ), nprocs=4, join=False,
      daemon=True, start_method=start_method)
  [monitor.add_pid(pid) for pid in ctx.pids()]

  try:
    for k in range(100):
      print(monitor.table())
      time.sleep(1)
  finally:
    ctx.join()
