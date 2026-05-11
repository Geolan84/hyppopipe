from __future__ import annotations

import logging
import sys
from typing import Final, TextIO

PACKAGE_ROOT: Final[str] = "hyppopipe"

_explicit_package_logging: bool = False


def ensure_default_logging(
    level: int | None = None,
    *,
    stream: TextIO | None = None,
) -> None:
    """
    If the package logger has no handlers, attach a stderr handler once.

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
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(handler)
    log.setLevel(effective)
    log.propagate = False


def configure_logging(
    level: int = logging.INFO,
    *,
    fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """
    Attach (or replace) a stderr handler on the ``hyppopipe`` logger.

    Call this once to take over logging for the package; automatic setup in
    :func:`ensure_default_logging` is then skipped.

    Use ``force=True`` to drop existing handlers on that logger and apply
    new formatting or level (for example in tests or a custom CLI).
    """
    global _explicit_package_logging
    _explicit_package_logging = True

    log = logging.getLogger(PACKAGE_ROOT)
    if force:
        for h in log.handlers[:]:
            log.removeHandler(h)
    elif log.handlers:
        log.setLevel(level)
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False
