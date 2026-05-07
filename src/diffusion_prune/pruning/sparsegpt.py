from ..logging import setup_logger
from .types import PruningConfig

logger = setup_logger(__name__)


def prune_with_sparsegpt(model, tokenizer, config: PruningConfig):
    logger.info(f"Applying SparseGPT pruning with sparsity={config.sparsity}")
    raise NotImplementedError("SparseGPT pruning not yet implemented")
