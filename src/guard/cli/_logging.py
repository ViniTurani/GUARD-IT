"""Redirect stdlib logging into loguru so all logs share the same format."""

import logging
import sys

from loguru import logger

__all__ = ["setup_logging"]


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru and route all stdlib logging through it."""
    logger.remove()
    logger.add(sys.stderr, level=level)

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
