import hashlib
import os
from pathlib import Path

from scripts.snapshot_and_upload_resume import (
    file_inventory,
    hardlink_or_copy,
    render_restore_document,
)


def test_hardlink_snapshot_and_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"checkpoint")
    destination = tmp_path / "bundle" / "checkpoint.bin"
    destination.parent.mkdir()

    hardlink_or_copy(str(source), str(destination))
    inventory = file_inventory(tmp_path / "bundle")

    assert destination.read_bytes() == b"checkpoint"
    assert os.stat(source).st_ino == os.stat(destination).st_ino
    assert inventory == {
        "checkpoint.bin": {
            "bytes": 10,
            "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        }
    }


def test_restore_document_records_exact_branch_and_roles() -> None:
    document = render_restore_document(
        "backup-id",
        "owner/repo",
        "backup/revision",
        "backup/branch",
        {"latest": {"step": 420}},
        2,
    )

    assert "owner/repo" in document
    assert "backup/revision" in document
    assert "backup/branch" in document
    assert "checkpoint-420" in document
    assert "world size 2" in document
