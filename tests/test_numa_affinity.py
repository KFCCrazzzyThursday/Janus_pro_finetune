from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


PLUGIN_PATH = Path(__file__).parents[1] / "training" / "plugins" / "numa_affinity.py"


def load_plugin_without_applying(monkeypatch):
    monkeypatch.delenv("JANUS_NUMA_CPUSETS", raising=False)
    monkeypatch.delenv("JANUS_NUMA_NODES", raising=False)
    spec = importlib.util.spec_from_file_location("numa_affinity_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_cpu_set(monkeypatch):
    plugin = load_plugin_without_applying(monkeypatch)
    assert plugin.parse_cpu_set("0-2,8,10-11") == {0, 1, 2, 8, 10, 11}


@pytest.mark.parametrize("spec", ["", "4-2", "-1"])
def test_parse_cpu_set_rejects_invalid_input(monkeypatch, spec):
    plugin = load_plugin_without_applying(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        plugin.parse_cpu_set(spec)


def test_rank_binding_stays_inside_existing_allowance(monkeypatch):
    plugin = load_plugin_without_applying(monkeypatch)
    original = os.sched_getaffinity(0)
    chosen = min(original)
    monkeypatch.setenv("JANUS_NUMA_CPUSETS", str(chosen))
    monkeypatch.setenv("JANUS_NUMA_NODES", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(plugin, "set_preferred_numa_node", lambda _node: True)
    try:
        rank, cpus, node, applied = plugin.apply_from_environment()
        assert (rank, cpus, node, applied) == (0, {chosen}, 0, True)
        assert os.sched_getaffinity(0) == {chosen}
    finally:
        os.sched_setaffinity(0, original)
