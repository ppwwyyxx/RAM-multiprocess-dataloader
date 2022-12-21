import pickle
import time
import torch
from common import MemoryMonitor, create_list

class NaiveDatasetFromList(torch.utils.data.Dataset):
  def __init__(self, lst):
    self.lst = lst
  def __len__(self):
    return len(self.lst)
  def __getitem__(self, idx: int):
    return self.lst[idx]


def worker(_, dataset: torch.utils.data.Dataset):
  while True:
    for sample in dataset:
      # read the data, with a fake latency
      time.sleep(0.000001)
      result = pickle.dumps(sample)


if __name__ == "__main__":
  monitor = MemoryMonitor()
  ds = NaiveDatasetFromList(create_list())
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
