import logging
import os

from rich.console import Console


def setup_logger(name: str = "diffusion_prune", level: str | None = None) -> logging.Logger:
    """Get a logger that uses Hydra's formatting.

    Logs will appear as: [timestamp][module][level] - message
    """
    if level is None:
        level = "DEBUG" if os.environ.get("HYDRA_FULL_ERROR", "0") == "1" else "INFO"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def get_console() -> Console:
    """Get a Rich console instance, creating it lazily to avoid pickling issues."""
    global _console
    if _console is None:
        _console = Console()
    return _console


_console: Console | None = None
