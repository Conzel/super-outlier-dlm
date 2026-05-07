from datetime import datetime

import numpy as np
from lm_eval import evaluator
from lm_eval.tasks import TaskManager

from ..logging import setup_logger
from .harness import create_harness
from .types import EvaluationConfig, EvaluationResult

logger = setup_logger(__name__)

# Default num_fewshot per task, used when config.num_fewshot is None.
DEFAULT_NUM_FEWSHOT = {
    "arc_challenge": 0,
    "arc_easy": 25,
    "boolq": 0,
    "hellaswag": 0,
    "openbookqa": 0,
    "piqa": 0,
    "winogrande": 5,
    "gsm8k": 4,
    "mmlu": 5,
    "mmlu_pro": 0,
}


def _get_primary_metric_for_task(task_name: str) -> str:
    """Get the primary accuracy metric name for a task from lm_eval's task config.

    lm_eval result keys follow the format:
    - "{metric},{filter_name}" for tasks with filters (e.g., exact_match,strict-match)
    - "{metric},none" for tasks without filters (e.g., acc,none)
    """
    tm = TaskManager()
    config = tm._get_config(task_name)

    metric_list = config.get("metric_list", [])
    filter_list = config.get("filter_list", [])

    if not metric_list:
        raise ValueError(f"Task has no associated metrics: {task_name}")

    primary_metric = metric_list[0].get("metric", "acc")

    # For tasks with filters, the key is "{metric},{filter_name}"
    if filter_list:
        filter_names = [f.get("name") for f in filter_list]
        if "strict-match" in filter_names:
            return f"{primary_metric},strict-match"
        return f"{primary_metric},{filter_names[0]}"

    # For tasks without filters, prefer acc_norm if available
    metric_names = [m.get("metric") for m in metric_list]
    if "acc_norm" in metric_names:
        return "acc_norm,none"

    return f"{primary_metric},none"


def _convert_numpy(obj):
    """Convert numpy types to native Python types for serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def evaluate_model(
    config: EvaluationConfig, model_config, pruning_config=None, quantization_config=None
):
    """Run LM evaluation harness on configured tasks."""
    task_list = config.task_list
    logger.info(f"Evaluating on task(s): {task_list}")
    logger.info(f"Model type: {model_config.model_type}")

    model_path = model_config.hf_model_name
    if model_config.model_base_path:
        model_path = f"{model_config.model_base_path}/{model_config.hf_model_name}"

    logger.info(f"Model path: {model_path}")

    # Create appropriate harness based on model type
    lm = create_harness(model_config.model_type, model_path, config, model_config=model_config)
    logger.info("Starting simple_evaluate...")

    try:
        if config.num_fewshot is not None:
            # Single num_fewshot for all tasks
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=task_list,
                batch_size=config.batch_size,
                num_fewshot=config.num_fewshot,
                limit=config.limit,
            )
        else:
            # Per-task defaults: run each task with its own num_fewshot
            results = None
            for task_name in task_list:
                fewshot = DEFAULT_NUM_FEWSHOT.get(task_name, 0)
                logger.info(f"Running {task_name} with num_fewshot={fewshot}")
                task_results = evaluator.simple_evaluate(
                    model=lm,
                    tasks=[task_name],
                    batch_size=config.batch_size,
                    num_fewshot=fewshot,
                    limit=config.limit,
                )
                if task_results is not None:
                    if results is None:
                        results = task_results
                    else:
                        for key in ("results", "versions", "n-shot", "configs", "higher_is_better"):
                            if key in task_results:
                                results[key].update(task_results[key])
                        if "n-samples" in task_results:
                            results.setdefault("n-samples", {}).update(task_results["n-samples"])
        logger.info(f"simple_evaluate returned: {results is not None}")
    except Exception as e:
        logger.error(f"Error in simple_evaluate: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise

    # In distributed mode, non-primary ranks get None
    if results is None:
        return []

    timestamp = datetime.now().isoformat()

    # Process results for each task
    evaluation_results = []
    for task_name in task_list:
        task_results = results["results"][task_name]

        # Get task-specific primary metric
        primary_metric = _get_primary_metric_for_task(task_name)
        accuracy = task_results.get(primary_metric)

        if accuracy is None:
            accuracy = 0.0
            logger.error(
                f"Metric '{primary_metric}' not found for task {task_name}. Available: {list(task_results.keys())}"
            )

        logger.info(f"Task {task_name}: {primary_metric} = {accuracy}")

        result = EvaluationResult(
            task=task_name,
            accuracy=float(accuracy),
            accuracy_metric=primary_metric,
            model_config=model_config,
            eval_config=config,
            pruning_config=pruning_config,
            quantization_config=quantization_config,
            timestamp=timestamp,
            additional_metrics=_convert_numpy(task_results),
        )
        evaluation_results.append(result)

    return evaluation_results
