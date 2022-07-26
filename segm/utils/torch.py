import os
import torch


"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(mode):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    
    gpu_id = [0,1,2,3]
    dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 4))

    distributed = world_size > 1
    use_gpu = mode
    dedvice = torch.device("cuda" if use_qpu else "cpu")
    torch.backends.cudnn.benchmark = True
