import gc
import torch
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class MemoryStatistics:
    peak_increased_gb: float | None = None


@contextmanager
def memory_monitor(device='cuda:0'):
    """Return the increased peak VRAM usage of inference (context range) in GB"""
    torch.cuda.empty_cache()
    gc.collect()
    recorder = MemoryStatistics()
    
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device)

    try:
        
        yield recorder
        
    finally:
        peak_memory = torch.cuda.max_memory_allocated(device)
        recorder.peak_increased_gb = (peak_memory - initial_memory) / (1024**3)
