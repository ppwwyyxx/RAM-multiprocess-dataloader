import os
import time
import itertools
import pickle
import multiprocessing as mp
import torch

import detectron2.utils.comm as comm
from detectron2.engine import launch

from detectron2.utils import logger
logger.setup_logger()

from common import MemoryMonitor, create_list, DatasetFromList
from serialize import TorchShmSerializedList

def worker(_, dataset: torch.utils.data.Dataset):
  while True:
    for sample in dataset:
      # read the data, with a fake latency
      time.sleep(0.000001)
      result = pickle.dumps(sample)


def main():
  monitor = MemoryMonitor()
  ds = DatasetFromList(TorchShmSerializedList(
      # Only GPU worker 0 needs to read data.
      create_list() if comm.get_local_rank() == 0 else None))
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
  mp.set_forkserver_preload(["torch"])
  num_gpus = 2
  if torch.cuda.device_count() < num_gpus:
    # We don't actually need GPUs anyway.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  # This uses "spawn" internally. To switch to forkserver, modifying
  # detectron2 source code is needed.
  launch(main, num_gpus, dist_url="auto")
