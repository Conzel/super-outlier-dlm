"""AlphaPruning: Layer-wise sparsity based on heavy-tailed self-regularization theory.

Based on "AlphaPruning: Using Heavy-Tailed Self Regularization Theory for Improved
Layer-wise Pruning of Large Language Models" (NeurIPS 2024).
https://github.com/haiquanlu/AlphaPruning

The method assigns different sparsity ratios to layers based on metrics derived from
the empirical spectral density (ESD) of weight matrices. Layers with higher metric
values (indicating better training) receive different pruning ratios than layers with
lower values.

Includes FARMS (Fixed Aspect Ratio Matrix Sampling) from AlphaQ for robust alpha
estimation on rectangular weight matrices.
"""

import random

import torch


def compute_layer_metric(
    weight: torch.Tensor,
    metric_type: str = "alpha_peak",
    use_farms: bool = True,
    farms_m_sub: int = 128,
    farms_n_sub: int = 128,
    farms_max_blocks: int = 256,
) -> float:
    """Compute a metric for a weight tensor that indicates layer trainability.

    Args:
        weight: Weight tensor to analyze.
        metric_type: Type of metric to compute. Options:
            - 'alpha_peak': Power-law exponent using histogram peak method (recommended)
            - 'alpha_mid': Power-law exponent using middle of spectrum
            - 'spectral_norm': Largest singular value
            - 'stable_rank': Frobenius norm squared / spectral norm squared
        use_farms: Use FARMS for robust alpha estimation (default: True, recommended).
        farms_m_sub: FARMS submatrix height (default: 128, from AlphaQ).
        farms_n_sub: FARMS submatrix width (default: 128, from AlphaQ).
        farms_max_blocks: Maximum FARMS blocks to sample (default: 256, from AlphaQ).

    Returns:
        Metric value (interpretation depends on metric_type).
    """
    if metric_type == "spectral_norm":
        return _compute_spectral_norm(weight)
    elif metric_type == "stable_rank":
        return _compute_stable_rank(weight)
    elif metric_type in ["alpha_peak", "alpha_mid"]:
        return _estimate_alpha(
            weight,
            method=metric_type,
            use_farms=use_farms,
            farms_m_sub=farms_m_sub,
            farms_n_sub=farms_n_sub,
            farms_max_blocks=farms_max_blocks,
        )
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def _compute_spectral_norm(weight: torch.Tensor) -> float:
    """Compute the spectral norm (largest singular value) of a weight tensor."""
    if weight.dim() == 2:
        s = torch.linalg.svdvals(weight.float())
    else:
        # For conv layers, flatten appropriately
        w_flat = weight.flatten(1).float()
        s = torch.linalg.svdvals(w_flat)
    return s[0].item()


def _compute_stable_rank(weight: torch.Tensor) -> float:
    """Compute the stable rank: Frobenius norm squared / spectral norm squared."""
    if weight.dim() == 2:
        s = torch.linalg.svdvals(weight.float())
    else:
        w_flat = weight.flatten(1).float()
        s = torch.linalg.svdvals(w_flat)

    spectral_norm_sq = (s[0] ** 2).item()
    frobenius_norm_sq = (s**2).sum().item()

    if spectral_norm_sq > 0:
        return frobenius_norm_sq / spectral_norm_sq
    return 1.0


def _compute_eigenvalues_baseline(weight: torch.Tensor) -> torch.Tensor:
    """Compute squared singular values (eigenvalues) using full matrix SVD."""
    if weight.dim() == 2:
        matrix = weight.float()
    else:
        matrix = weight.flatten(1).float()

    matrix = matrix.cpu()

    try:
        s = torch.linalg.svdvals(matrix)
        eigs = s**2
        eigs, _ = torch.sort(eigs, descending=False)
        return eigs
    except Exception:
        return torch.tensor([], dtype=torch.float32)


def _compute_eigenvalues_farms(
    weight: torch.Tensor,
    m_sub: int = 128,
    n_sub: int = 128,
    stride_m: int = 128,
    stride_n: int = 128,
    max_blocks: int = 256,
    seed: int | None = None,
) -> torch.Tensor:
    """Compute eigenvalues using FARMS (Fixed Aspect Ratio Matrix Sampling).

    FARMS samples fixed-size submatrices to compute more robust alpha estimates
    on rectangular weight matrices with extreme aspect ratios.

    Args:
        weight: Weight tensor to analyze.
        m_sub: Height of sampled submatrices.
        n_sub: Width of sampled submatrices.
        stride_m: Row stride for sampling.
        stride_n: Column stride for sampling.
        max_blocks: Maximum number of submatrices to sample.
        seed: Random seed for sampling (None = no seeding).

    Returns:
        Concatenated and sorted eigenvalues from all sampled submatrices.
    """
    if weight.dim() == 2:
        matrix = weight.float()
    else:
        matrix = weight.flatten(1).float()

    m, n = matrix.shape

    # If submatrix size is larger than matrix, fall back to baseline
    if m_sub > m or n_sub > n:
        return _compute_eigenvalues_baseline(weight)

    matrix = matrix.cpu()

    # Generate all possible block indices
    indices = []
    for i in range(0, m - m_sub + 1, max(1, stride_m)):
        for j in range(0, n - n_sub + 1, max(1, stride_n)):
            indices.append((i, j))

    if len(indices) == 0:
        return _compute_eigenvalues_baseline(weight)

    # Randomly sample if too many blocks
    if seed is not None:
        random.seed(seed)
    if len(indices) > max_blocks:
        indices = random.sample(indices, max_blocks)

    # Compute eigenvalues for each sampled block
    eig_list = []
    for i, j in indices:
        sub = matrix[i : i + m_sub, j : j + n_sub]
        s = torch.linalg.svdvals(sub)
        eigs = s**2
        eig_list.append(eigs)

    if not eig_list:
        return torch.tensor([], dtype=torch.float32)

    # Concatenate and sort all eigenvalues
    eigs_all = torch.cat(eig_list, dim=0)
    eigs_all, _ = torch.sort(eigs_all, descending=False)
    return eigs_all


def _estimate_alpha(
    weight: torch.Tensor,
    method: str = "alpha_peak",
    use_farms: bool = True,
    farms_m_sub: int = 128,
    farms_n_sub: int = 128,
    farms_max_blocks: int = 256,
) -> float:
    """Estimate the alpha parameter from power-law fitting of eigenvalues.

    This implements the core ESD analysis from AlphaPruning. The alpha parameter
    characterizes the heavy-tailed distribution of eigenvalues, with lower alpha
    indicating heavier tails and better-trained layers.

    Args:
        weight: Weight tensor to analyze.
        method: Method for selecting xmin cutoff:
            - 'alpha_peak': Use histogram peak to find xmin (recommended)
            - 'alpha_mid': Use middle of spectrum as xmin
        use_farms: Use FARMS (Fixed Aspect Ratio Matrix Sampling, default: True).
        farms_m_sub: FARMS submatrix height (default: 128, from AlphaQ).
        farms_n_sub: FARMS submatrix width (default: 128, from AlphaQ).
        farms_max_blocks: Maximum FARMS blocks to sample (default: 256, from AlphaQ).

    Returns:
        Estimated alpha value (typically in range 1.0-5.0).
    """
    # Compute eigenvalues using FARMS or baseline method
    if use_farms:
        eigs = _compute_eigenvalues_farms(
            weight,
            m_sub=farms_m_sub,
            n_sub=farms_n_sub,
            stride_m=farms_m_sub,  # Non-overlapping by default
            stride_n=farms_n_sub,
            max_blocks=farms_max_blocks,
        )
    else:
        eigs = _compute_eigenvalues_baseline(weight)

    if len(eigs) == 0:
        return 2.0

    # Keep all eigenvalues (reference AlphaPruning uses filter_zeros=False by default)
    nz_eigs = eigs
    N = len(nz_eigs)
    log_nz_eigs = torch.log(nz_eigs)

    if method == "alpha_mid":
        # Use middle of spectrum as xmin
        i = N // 2
        n = float(N - i)
        if n > 0:
            alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            return max(alpha.item(), 1.0)
        return 2.0

    elif method == "alpha_peak":
        # Use histogram peak to find xmin, then optimize via KS statistic
        # (matches reference AlphaPruning implementation)
        hist_nz_eigs = torch.log10(nz_eigs + 1e-10)
        min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()

        if max_e - min_e < 1e-6:
            return 2.0

        bins = min(100, N // 2)
        counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
        boundaries = torch.linspace(min_e, max_e, bins + 1)

        ih = torch.argmax(counts)
        xmin2 = 10 ** boundaries[ih]

        # Define search range around the peak: [0.95 * peak, 1.5 * peak]
        # NOTE: the reference AlphaPruning has a bug here where xmin_min is in
        # log10 space but compared against linear eigenvalues, making the lower
        # bound a no-op. We use consistent linear-scale bounds instead.
        xmin_min = 0.95 * xmin2
        xmin_max = 1.5 * xmin2

        # Search over candidate xmin values, pick the one minimizing KS distance
        alphas = torch.zeros(N - 1)
        Ds = torch.ones(N - 1)

        for i, xmin in enumerate(nz_eigs[:-1]):
            if xmin < xmin_min:
                continue
            if xmin > xmin_max:
                break

            n = float(N - i)
            seq = torch.arange(n)
            alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            alphas[i] = alpha
            if alpha > 1:
                Ds[i] = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n))

        min_D_index = torch.argmin(Ds)
        final_alpha = alphas[min_D_index].item()

        return max(final_alpha, 1.0)

    return 2.0


def compute_alpha_pruning_ratios(
    weights: list[torch.Tensor],
    target_sparsity: float,
    epsilon: float = 0.15,
    metric_type: str = "alpha_peak",
    layer_metrics: list[float] | None = None,
    use_farms: bool = True,
    farms_m_sub: int = 128,
    farms_n_sub: int = 128,
    farms_max_blocks: int = 256,
) -> list[float]:
    """Compute AlphaPruning layer-wise sparsity ratios for all layers.

    This implements the core AlphaPruning algorithm:
    1. Compute or load metrics for each layer
    2. Normalize metrics to [0, 1]
    3. Map to sparsity range [target-epsilon, target+epsilon]
    4. Shift to achieve target overall sparsity (weighted by param count)

    Args:
        weights: List of weight tensors for all layers.
        target_sparsity: Overall target sparsity ratio (0.0 to 1.0).
        epsilon: Additive range for per-layer sparsity variation.
            Sparsity varies in [target-epsilon, target+epsilon].
        metric_type: Type of metric to use. Default: 'alpha_peak'.
        layer_metrics: Pre-computed metrics for all layers. If None, computed from weights.
        use_farms: Use FARMS for robust estimation (default: True, from AlphaQ).
        farms_m_sub: FARMS submatrix height (default: 128, from AlphaQ).
        farms_n_sub: FARMS submatrix width (default: 128, from AlphaQ).
        farms_max_blocks: Maximum FARMS blocks to sample (default: 256, from AlphaQ).

    Returns:
        List of sparsity ratios, one per layer. Individual ratios may be outside
        [0, 1] range, but overall sparsity will match target_sparsity.

    Example:
        >>> # Compute metrics for all layers
        >>> weights = [layer.weight for layer in model.layers]
        >>> ratios = compute_alpha_pruning_ratios(
        ...     weights, target_sparsity=0.5, epsilon=0.15, use_farms=True
        ... )
        >>> # Apply pruning with computed ratios
        >>> for i, layer in enumerate(model.layers):
        ...     prune_layer(layer, ratios[i])
    """
    num_layers = len(weights)

    # Compute or use provided metrics
    if layer_metrics is None:
        mode_str = "FARMS" if use_farms else "baseline"
        print(f"Computing layer metrics for AlphaPruning ({mode_str} mode)...")
        metrics = []
        for i, w in enumerate(weights):
            metric = compute_layer_metric(
                w,
                metric_type,
                use_farms=use_farms,
                farms_m_sub=farms_m_sub,
                farms_n_sub=farms_n_sub,
                farms_max_blocks=farms_max_blocks,
            )
            metrics.append(metric)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_layers} layers")
    else:
        metrics = layer_metrics

    print(f"Layer metrics ({metric_type}): {metrics}")

    metrics_tensor = torch.tensor(metrics, dtype=torch.float32)

    # Normalize metrics to [0, 1]
    max_metric = metrics_tensor.max()
    min_metric = metrics_tensor.min()

    if max_metric - min_metric < 1e-6:
        # All metrics are the same, use uniform sparsity
        print("All metrics are identical, using uniform sparsity")
        return [target_sparsity] * num_layers

    normalized = (metrics_tensor - min_metric) / (max_metric - min_metric)

    # Map to [target-epsilon, target+epsilon] range
    layerwise_sparsities = normalized * (2 * epsilon) + (target_sparsity - epsilon)

    # Compute total number of parameters per layer
    num_params = torch.tensor([w.numel() for w in weights], dtype=torch.float32)

    # Shift to achieve target overall sparsity (weighted by param count)
    weighted_mean = (num_params * layerwise_sparsities).sum() / num_params.sum()
    final_ratios = layerwise_sparsities + (target_sparsity - weighted_mean)

    print(f"AlphaPruning ratios (min={final_ratios.min():.3f}, max={final_ratios.max():.3f}):")
    print(f"  {final_ratios.tolist()}")

    return final_ratios.tolist()


def precompute_alpha_pruning_for_model(
    model,
    target_sparsity: float,
    epsilon: float = 0.15,
    metric_type: str = "alpha_peak",
    use_farms: bool = True,
) -> dict[tuple[int, str], float]:
    """Pre-compute per-sublayer alpha pruning ratios for all linear modules.

    Must be called before the pruning loop. Each sublayer (q_proj, k_proj, etc.)
    gets its own alpha metric and sparsity ratio.

    Args:
        model: The model to compute ratios for.
        target_sparsity: Overall target sparsity (0-1).
        epsilon: Range parameter for sparsity variation.
        metric_type: Alpha metric type (default: alpha_peak).
        use_farms: Use FARMS for robust alpha estimation (default: True).

    Returns:
        Dict mapping (block_idx, sublayer_name) to sparsity ratio.
    """
    from ..model.utils import get_model_layers
    from .magnitude import find_layers
    from .sparsity_strategy import _alpha_pruning_state

    _alpha_pruning_state.reset()

    layers = get_model_layers(model)
    all_weights = []
    all_keys = []

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        for name, linear in subset.items():
            all_weights.append(linear.weight.data)
            all_keys.append((i, name))

    ratios_list = compute_alpha_pruning_ratios(
        weights=all_weights,
        target_sparsity=target_sparsity,
        epsilon=epsilon,
        metric_type=metric_type,
        use_farms=use_farms,
    )

    ratios = dict(zip(all_keys, ratios_list, strict=False))
    _alpha_pruning_state.set_ratios(ratios)
    return ratios


def load_precomputed_metrics(metrics_file: str) -> list[float]:
    """Load pre-computed layer metrics from a file.

    Args:
        metrics_file: Path to .npy file containing metrics.

    Returns:
        List of metric values, one per layer.
    """
    import numpy as np

    metrics = np.load(metrics_file)
    return metrics.tolist()


def save_metrics(metrics: list[float], output_file: str) -> None:
    """Save computed metrics to a file for later reuse.

    Args:
        metrics: List of metric values.
        output_file: Path to save .npy file.
    """
    import numpy as np

    np.save(output_file, np.array(metrics))
    print(f"Saved metrics to {output_file}")
