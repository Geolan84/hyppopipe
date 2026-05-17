from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hyppopipe.package_logging import (
    PACKAGE_ROOT,
    LogConfig,
    configure_logging,
    reset_package_logging,
    run_logging,
)


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    reset_package_logging()
    yield
    reset_package_logging()


def test_run_logging_writes_file(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "train.log"
    child = logging.getLogger(f"{PACKAGE_ROOT}.train.trainer")

    with run_logging(log_file):
        child.info("epoch 1")

    text = log_file.read_text(encoding="utf-8")
    assert "epoch 1" in text
    assert not logging.getLogger(PACKAGE_ROOT).handlers


def test_run_logging_log_config_console_and_file(tmp_path: Path) -> None:
    log_file = tmp_path / "train.log"
    child = logging.getLogger(f"{PACKAGE_ROOT}.pipeline.pipeline")

    with run_logging(LogConfig(file=log_file, console=True)):
        child.warning("both sinks")

    assert "both sinks" in log_file.read_text(encoding="utf-8")


def test_configure_logging_file(tmp_path: Path) -> None:
    log_file = tmp_path / "global.log"
    configure_logging(log_file=log_file, console=False)
    logging.getLogger(f"{PACKAGE_ROOT}.train.trainer").error("persistent")

    assert "persistent" in log_file.read_text(encoding="utf-8")


def test_run_logging_stacks_on_configure_logging(tmp_path: Path) -> None:
    global_log = tmp_path / "global.log"
    run_log = tmp_path / "run.log"
    configure_logging(log_file=global_log, console=False)
    package_log = logging.getLogger(PACKAGE_ROOT)

    with run_logging(run_log):
        assert len(package_log.handlers) == 2
        package_log.info("run message")

    assert "run message" in run_log.read_text(encoding="utf-8")
    assert "run message" in global_log.read_text(encoding="utf-8")
    assert len(package_log.handlers) == 1
