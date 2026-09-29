"""Host commit telemetry and conservative admission/stop gates for Windows studies."""
import ctypes as ct
from ctypes import wintypes as wt
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

COMMIT_RESERVE = 8 * 1024**3
WORKER_START_ALLOWANCE = 8 * 1024**3


class ResourcePressureError(RuntimeError):
    """Resource reserve reached; the trial is incomplete, not a learning failure."""


class _Performance(ct.Structure):
    _fields_ = [('cb', wt.DWORD)] + [(name, ct.c_size_t) for name in (
        'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
        'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize',
    )] + [(name, wt.DWORD) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]


class _ProcessMemory(ct.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + [
        (name, ct.c_size_t) for name in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
            'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage',
        )
    ]


@lru_cache(maxsize=1)
def _windows_api():
    if sys.platform != 'win32':
        raise OSError('Monitored study resource telemetry currently requires Windows')
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wt.HANDLE
    api = ct.WinDLL('psapi', use_last_error=True)
    api.GetPerformanceInfo.argtypes = [ct.POINTER(_Performance), wt.DWORD]
    api.GetPerformanceInfo.restype = wt.BOOL
    api.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ct.POINTER(_ProcessMemory), wt.DWORD]
    api.GetProcessMemoryInfo.restype = wt.BOOL
    return kernel, api


def host_snapshot():
    kernel, api = _windows_api()
    system = _Performance(cb=ct.sizeof(_Performance))
    process = _ProcessMemory(cb=ct.sizeof(_ProcessMemory))
    if not api.GetPerformanceInfo(ct.byref(system), ct.sizeof(system)):
        raise ct.WinError(ct.get_last_error())
    if not api.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ct.byref(process), ct.sizeof(process)):
        raise ct.WinError(ct.get_last_error())
    return {
        'host_budget_kind': 'windows_commit',
        'commit_total_bytes': system.CommitTotal * system.PageSize,
        'commit_limit_bytes': system.CommitLimit * system.PageSize,
        'commit_peak_bytes': system.CommitPeak * system.PageSize,
        'commit_available_bytes': (system.CommitLimit - system.CommitTotal) * system.PageSize,
        'physical_available_bytes': system.PhysicalAvailable * system.PageSize,
        'process_private_bytes': process.PrivateUsage,
        'process_rss_bytes': process.WorkingSetSize,
        'process_peak_private_bytes': process.PeakPagefileUsage,
    }


class ResourceMonitor:
    def __init__(self, path, *, device='cpu'):
        self.path, self.device = Path(path), device
        self.path.touch(exist_ok=False)
        self.latest = None
        self.last_sample = float('-inf')

    def check(self, phase, *, admission_slots=0, force=False):
        now = time.monotonic()
        if not force and not admission_slots and now - self.last_sample < 1:
            return
        row = dict(host_snapshot(), phase=phase, pid=os.getpid(), time=time.time())
        torch = sys.modules.get('torch')
        if self.device == 'cuda' and torch is not None and torch.cuda.is_initialized():
            row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                       cuda_reserved_bytes=torch.cuda.memory_reserved(),
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        required = COMMIT_RESERVE + admission_slots * WORKER_START_ALLOWANCE
        row['required_commit_available_bytes'] = required
        self.latest, self.last_sample = row, now
        with self.path.open('a', encoding='utf-8') as stream:
            json.dump(row, stream, allow_nan=False)
            stream.write('\n')
            stream.flush()
        if row['commit_available_bytes'] < required:
            raise ResourcePressureError(
                f"{phase}: host commit headroom {row['commit_available_bytes']:,} bytes "
                f"is below required reserve {required:,}; telemetry: {self.path}")
