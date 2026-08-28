from datasets import Dataset

from swift.dataset.loader import _inject_dataset_routing_tag


def test_existing_dataset_routing_tag_is_preserved() -> None:
    source = Dataset.from_dict({"id": ["one", "two"], "dataset": ["tqa", "tqa"]})

    routed = _inject_dataset_routing_tag(source, "/path/to/source.jsonl")

    assert routed.column_names.count("dataset") == 1
    assert routed["dataset"] == ["tqa", "tqa"]


def test_missing_dataset_routing_tag_is_injected() -> None:
    source = Dataset.from_dict({"id": ["one", "two"]})

    routed = _inject_dataset_routing_tag(source, "source-name")

    assert routed["dataset"] == ["source-name", "source-name"]
