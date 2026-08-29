"""Apply per-rank CPU and preferred-memory NUMA affinity before model load."""

from __future__ import annotations

import ctypes
import os

from swift.utils import get_logger


logger = get_logger()


def parse_cpu_set(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first_raw, last_raw = part.split("-", 1)
            first, last = int(first_raw), int(last_raw)
            if first < 0 or last < first:
                raise ValueError(f"Invalid CPU range: {part}")
            cpus.update(range(first, last + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"Invalid CPU index: {part}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set must not be empty")
    return cpus


def set_preferred_numa_node(node: int) -> bool:
    """Prefer ``node`` for future allocations while allowing fallback."""
    try:
        libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError:
        return False
    libnuma.numa_available.restype = ctypes.c_int
    if libnuma.numa_available() < 0:
        return False
    libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
    libnuma.numa_set_preferred.restype = None
    libnuma.numa_set_preferred(node)
    return True


def apply_from_environment() -> tuple[int, set[int], int, bool] | None:
    cpu_specs = os.environ.get("JANUS_NUMA_CPUSETS")
    node_specs = os.environ.get("JANUS_NUMA_NODES")
    if not cpu_specs or not node_specs:
        return None

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cpu_sets = cpu_specs.split(";")
    nodes = [int(item) for item in node_specs.split(",")]
    if len(cpu_sets) != len(nodes):
        raise ValueError("JANUS_NUMA_CPUSETS and JANUS_NUMA_NODES lengths differ")
    if not 0 <= local_rank < len(cpu_sets):
        raise ValueError(f"LOCAL_RANK={local_rank} has no NUMA mapping")

    requested_cpus = parse_cpu_set(cpu_sets[local_rank])
    allowed_cpus = os.sched_getaffinity(0)
    selected_cpus = requested_cpus & allowed_cpus
    if not selected_cpus:
        raise RuntimeError(
            f"Rank {local_rank} NUMA CPU set {sorted(requested_cpus)} does not overlap "
            f"the process allowance {sorted(allowed_cpus)}"
        )
    os.sched_setaffinity(0, selected_cpus)
    node = nodes[local_rank]
    preferred_set = set_preferred_numa_node(node)
    return local_rank, selected_cpus, node, preferred_set


_result = apply_from_environment()
if _result is not None:
    _rank, _cpus, _node, _preferred_set = _result
    logger.info(
        "NUMA affinity: local_rank=%d CPUs=%s preferred_node=%d memory_policy_applied=%s",
        _rank,
        sorted(_cpus),
        _node,
        _preferred_set,
    )
