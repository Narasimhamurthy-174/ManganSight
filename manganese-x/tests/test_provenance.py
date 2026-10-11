from src.common.config import load_config, resolve
from src.common.provenance import load_registry, validate_registry


def test_project_registry_is_valid():
    cfg = load_config()
    assert validate_registry(load_registry(resolve(cfg["paths"]["registry"]))) == []


def test_missing_field_and_bad_class_detected():
    probs = validate_registry([{"dataset_id": "X", "data_class": "MAGIC"}])
    assert any("missing field" in p for p in probs)
    assert any("data_class" in p for p in probs)
