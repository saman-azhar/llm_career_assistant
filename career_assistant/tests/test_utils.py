# test_utils.py
import os
import json
import pandas as pd
import pytest
from pathlib import Path
from career_assistant.utils import file_utils, logger

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

# -------------------------
# file_utils.py tests
# -------------------------
def test_read_write_csv(tmp_dir):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_dir / "test.csv"

    # write
    file_utils.write_csv(df, csv_path)
    assert csv_path.exists()

    # read
    df_read = file_utils.read_csv(csv_path)
    pd.testing.assert_frame_equal(df, df_read)

def test_read_write_json(tmp_dir):
    data = {"key": "value", "list": [1, 2, 3]}
    json_path = tmp_dir / "test.json"

    # write
    file_utils.write_json(data, json_path)
    assert json_path.exists()

    # read
    data_read = file_utils.read_json(json_path)
    assert data_read == data

def test_ensure_dir_creates_directory(tmp_dir):
    new_dir = tmp_dir / "nested/dir"
    returned_path = file_utils.ensure_dir(new_dir)
    assert new_dir.exists() and returned_path == new_dir

def test_read_nonexistent_csv_raises(tmp_dir):
    with pytest.raises(FileNotFoundError):
        file_utils.read_csv(tmp_dir / "nonexistent.csv")

def test_read_nonexistent_json_raises(tmp_dir):
    with pytest.raises(FileNotFoundError):
        file_utils.read_json(tmp_dir / "nonexistent.json")

# -------------------------
# logger.py tests
# -------------------------
def test_get_logger_returns_logger():
    log = logger.get_logger("test_logger")
    assert log.name == "test_logger"
    assert hasattr(log, "info")

def test_logger_writes_to_file(tmp_dir, monkeypatch):
    log_file = tmp_dir / "test.log"
    monkeypatch.setattr(logger, "LOG_FILE", str(log_file))

    log = logger.get_logger("file_test_logger")
    log.info("Test message")
    log.warning("Warning message")
    log.error("Error message")

    # flush handlers
    for h in log.handlers:
        h.flush()

    # Check file content
    with open(log_file, "r") as f:
        content = f.read()
        assert "Test message" in content
        assert "Warning message" in content
        assert "Error message" in content

def test_logger_no_duplicate_handlers(tmp_dir, monkeypatch):
    log_file = tmp_dir / "test.log"
    monkeypatch.setattr(logger, "LOG_FILE", str(log_file))

    log1 = logger.get_logger("dup_logger")
    log2 = logger.get_logger("dup_logger")
    # Only one file and console handler should exist
    handler_types = [type(h) for h in log1.handlers]
    assert handler_types.count(logger.RotatingFileHandler) == 1
    assert handler_types.count(logger.StreamHandler) == 1
