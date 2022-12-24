#!/usr/bin/env python
import itertools
import multiprocessing as mp
import os
import pickle
import time
import torch

import detectron2.utils.comm as comm
from detectron2.engine import launch

from common import MemoryMonitor, create_list, DatasetFromList
from serialize import TorchSerializedList

def worker(_, dataset: torch.utils.data.Dataset):
  while True:
    for sample in dataset:
      # read the data, with a fake latency
      time.sleep(0.000001)
      result = pickle.dumps(sample)


def main():
  monitor = MemoryMonitor()
  ds = DatasetFromList(TorchSerializedList(create_list()))
  print(monitor.table())

  mp.set_forkserver_preload(["torch"])
  ctx = torch.multiprocessing.start_processes(
      worker, (ds, ), nprocs=4, join=False, daemon=True,
      start_method='forkserver')

  all_pids = comm.all_gather([os.getpid()] + ctx.pids())
  all_pids = list(itertools.chain.from_iterable(all_pids))
  monitor = MemoryMonitor(all_pids)

  try:
    for k in range(100):
      # Print memory (of all processes) in the main process only.
      if comm.is_main_process():
        print(monitor.table())
      time.sleep(1)
  finally:
    ctx.join()

if __name__ == "__main__":
  num_gpus = 2
  if torch.cuda.device_count() < num_gpus:
    # We don't actually need GPUs anyway.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  launch(main, num_gpus, dist_url="auto")
