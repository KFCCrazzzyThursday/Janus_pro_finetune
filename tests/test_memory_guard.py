from __future__ import annotations

import importlib.util
import os
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "memory_guard.py"


def load_guard_module():
    spec = importlib.util.spec_from_file_location("memory_guard_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_memory_reading_is_sane():
    guard = load_guard_module()
    used, available, swap_used = guard.read_memory_gib()
    assert used >= 0
    assert available > 0
    assert swap_used >= 0


def test_process_tree_excludes_guard_pid():
    guard = load_guard_module()
    descendants = guard.process_tree(os.getppid(), exclude={os.getpid()})
    assert os.getpid() not in descendants
