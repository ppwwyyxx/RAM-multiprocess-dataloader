#!/usr/bin/env python
import time
import torch

from common import MemoryMonitor, create_coco, read_sample
from common import DatasetFromList as NaiveDatasetFromList


def worker(_, dataset: torch.utils.data.Dataset):
  while True:
    for sample in dataset:
      # read the data, with a fake latency
      time.sleep(0.000001)
      result = read_sample(sample)


if __name__ == "__main__":
  monitor = MemoryMonitor()
  ds = NaiveDatasetFromList(create_coco())
  print(monitor.table())

  ctx = torch.multiprocessing.start_processes(
      worker, (ds, ), nprocs=4, join=False,
      daemon=True, start_method='fork')
  [monitor.add_pid(pid) for pid in ctx.pids()]

  try:
    for k in range(100):
      print(monitor.table())
      time.sleep(1)
  finally:
    ctx.join()
