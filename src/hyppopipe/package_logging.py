"""Centralized logging configuration for the hyppopipe package."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

PACKAGE_ROOT: Final[str] = "hyppopipe"

_DEFAULT_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

_explicit_package_logging: bool = False


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Per-run logging options for :func:`run_logging`."""

    level: int = logging.INFO
    console: bool = True
    file: Path | str | None = None
    fmt: str = _DEFAULT_FMT
    datefmt: str = _DEFAULT_DATEFMT


def _coerce_log_config(log: Path | str | LogConfig) -> LogConfig:
    """Normalize a path or config into :class:`LogConfig`."""
    if isinstance(log, LogConfig):
        return log
    return LogConfig(file=log, console=False)


def _formatter(*, fmt: str, datefmt: str) -> logging.Formatter:
    """Build a :class:`logging.Formatter` with the given format strings."""
    return logging.Formatter(fmt=fmt, datefmt=datefmt)


def _attach_handlers(
    log: logging.Logger,
    handlers: list[logging.Handler],
    *,
    level: int,
) -> None:
    """Register handlers on ``log`` and disable propagation to the root logger."""
    for handler in handlers:
        handler.setLevel(level)
        log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False


def _build_handlers(
    *,
    level: int,
    fmt: str,
    datefmt: str,
    stream: TextIO | None,
    log_file: Path | str | None,
    console: bool,
) -> list[logging.Handler]:
    """Create stderr and/or file handlers for a logging run."""
    formatter = _formatter(fmt=fmt, datefmt=datefmt)
    handlers: list[logging.Handler] = []
    if console:
        stream_handler = logging.StreamHandler(stream or sys.stderr)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    if not handlers:
        msg = "At least one of console=True or log_file must be enabled"
        raise ValueError(msg)
    return handlers


def ensure_default_logging(
    level: int | None = None,
    *,
    stream: TextIO | None = None,
) -> None:
    """If the package logger has no handlers, attach a stderr handler once.

    Does nothing after :func:`configure_logging` has been called (explicit
    configuration replaces the default).
    """
    if _explicit_package_logging:
        return

    log = logging.getLogger(PACKAGE_ROOT)
    if log.handlers:
        if level is not None:
            log.setLevel(level)
        return

    effective = level if level is not None else logging.INFO
    handlers = _build_handlers(
        level=effective,
        fmt=_DEFAULT_FMT,
        datefmt=_DEFAULT_DATEFMT,
        stream=stream,
        log_file=None,
        console=True,
    )
    _attach_handlers(log, handlers, level=effective)


def configure_logging(
    level: int = logging.INFO,
    *,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
    stream: TextIO | None = None,
    log_file: Path | str | None = None,
    console: bool = True,
    force: bool = False,
) -> None:
    """Attach handlers on the ``hyppopipe`` logger (stderr and/or file).

    Call this once to take over logging for the package; automatic setup in
    :func:`ensure_default_logging` is then skipped.

    Use ``force=True`` to drop existing handlers on that logger and apply
    new formatting or level (for example in tests or a custom CLI).
    """
    global _explicit_package_logging
    _explicit_package_logging = True

    log = logging.getLogger(PACKAGE_ROOT)
    if force:
        for handler in log.handlers[:]:
            log.removeHandler(handler)
            handler.close()
    elif log.handlers:
        log.setLevel(level)
        return

    handlers = _build_handlers(
        level=level,
        fmt=fmt,
        datefmt=datefmt,
        stream=stream,
        log_file=log_file,
        console=console,
    )
    _attach_handlers(log, handlers, level=level)


@contextmanager
def run_logging(log: Path | str | LogConfig | None = None) -> Iterator[None]:
    """Per-run logging for training or other jobs.

    * ``None`` — same as :func:`ensure_default_logging` for this block.
    * path — write only to that file (no console) for the block.
    * :class:`LogConfig` — full control (console and/or file).

    Handlers added here are removed when the block exits. Does not remove
    handlers installed by :func:`configure_logging`.
    """
    if log is None:
        ensure_default_logging()
        yield
        return

    config = _coerce_log_config(log)
    if not config.console and config.file is None:
        msg = "LogConfig must enable console and/or file"
        raise ValueError(msg)

    package_log = logging.getLogger(PACKAGE_ROOT)
    run_handlers = _build_handlers(
        level=config.level,
        fmt=config.fmt,
        datefmt=config.datefmt,
        stream=None,
        log_file=config.file,
        console=config.console,
    )
    _attach_handlers(package_log, run_handlers, level=config.level)
    try:
        yield
    finally:
        for handler in run_handlers:
            package_log.removeHandler(handler)
            handler.close()


def reset_package_logging() -> None:
    """Clear package logger handlers and explicit-config flag (tests)."""
    global _explicit_package_logging
    _explicit_package_logging = False
    log = logging.getLogger(PACKAGE_ROOT)
    for handler in log.handlers[:]:
        log.removeHandler(handler)
        handler.close()
