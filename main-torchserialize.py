#!/usr/bin/env python
import pickle
import sys
import time
import torch
import multiprocessing as mp

from common import MemoryMonitor, create_list, DatasetFromList
from serialize import TorchSerializedList


def worker(_, dataset: torch.utils.data.Dataset):
  while True:
    for sample in dataset:
      # read the data, with a fake latency
      time.sleep(0.000001)
      result = pickle.dumps(sample)


if __name__ == "__main__":
  start_method = sys.argv[1]
  monitor = MemoryMonitor()
  ds = DatasetFromList(TorchSerializedList(create_list()))
  print(monitor.table())
  if start_method == "forkserver":
    # Reduce 150M-per-process USS due to "import torch".
    mp.set_forkserver_preload(["torch"])

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
