"""Logging filter that adds distributed rank information to log records."""

import logging
import os


class AddRankInfo(logging.Filter):
    """Adds rank and world_size attributes to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        world_size = os.environ.get("WORLD_SIZE", "1")
        record.rank = str(int(rank) + 1)  # to e.g. make it display 1/2
        record.world_size = world_size
        return True


class OnlyFirstGPU(AddRankInfo):
    """Adds rank and world_size attributes to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        super().filter(record)  # side effect: adds rank and world_size attributes
        if int(record.rank) == 1:
            return True
        else:
            return False
