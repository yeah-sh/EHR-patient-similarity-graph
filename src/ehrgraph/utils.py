"""Shared helpers: reproducible seeding and logging setup."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducible runs.

    Also forces deterministic cuDNN kernels. ``PYTHONHASHSEED`` is exported so
    that subprocesses inherit it; it cannot change hashing in the running process.

    Args:
        seed: The seed value, normally ``Config.seed``.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a named logger, installing a console handler on the root logger once.

    Args:
        name: Logger name, usually ``__name__``.
        level: Logging level for this logger (and for the root handler on first call).

    Returns:
        The configured logger.
    """
    logging.basicConfig(format=LOG_FORMAT, level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
