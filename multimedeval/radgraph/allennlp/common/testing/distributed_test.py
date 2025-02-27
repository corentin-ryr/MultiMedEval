import datetime
from typing import List, Dict, Any, Tuple, Callable
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multimedeval.radgraph.allennlp.common.checks import check_for_gpu


def init_process(
    process_rank: int,
    distributed_device_ids: List[int] = None,
    world_size: int = 1,
    func: Callable = None,
    func_args: Tuple = None,
    func_kwargs: Dict[str, Any] = None,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
):
    assert world_size > 1

    global_rank = process_rank

    gpu_id = distributed_device_ids[process_rank]  # type: ignore

    if gpu_id >= 0:
        torch.cuda.set_device(int(gpu_id))
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=global_rank,
        )
    else:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=global_rank,
            timeout=datetime.timedelta(seconds=120),
        )

    func(global_rank, world_size, gpu_id, *func_args, **func_kwargs)

    dist.barrier()


def run_distributed_test(
    device_ids: List[int] = [-1, -1],
    func: Callable = None,
    *args,
    **kwargs,
):
    """
    This runs the `func` in a simulated distributed environment.

    # Parameters

    device_ids: `List[int]`
        List of devices. There need to be at least 2 devices. Default is [-1, -1].

    func: `Callable`
        `func` needs to be global for spawning the processes, so that it can be pickled.
    """
    check_for_gpu(device_ids)
    # "fork" start method is the default and should be preferred, except when we're
    # running the tests on GPU, in which case we need to use "spawn".
    start_method = "spawn" if any(x >= 0 for x in device_ids) else "fork"
    nprocs = world_size = len(device_ids)
    mp.start_processes(
        init_process,
        args=(device_ids, world_size, func, args, kwargs),
        nprocs=nprocs,
        start_method=start_method,
    )
