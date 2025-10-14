from types import SimpleNamespace
from typing import Optional, Union

import torch


_sampling_module = None

def top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for top-k and top-p sampling from pre-softmax logits,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    logits: torch.Tensor
        Pre-softmax logits for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of logits. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    filter_apply_order: str
        The order of applying top-k and top-p sampling, should be either ``"top_k_first"`` or ``"joint"``.
        If ``"top_k_first"``, we first apply top-k filter, then apply top-p sampling on the top-k results.
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round. Default is ``"top_k_first"``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = 0.5
    >>> top_k = 3
    >>> logits = torch.rand(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[ 1.9269,  1.4873,  0.9007, -2.1055, -0.7581],
            [ 1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
            [-0.4934,  0.2415, -0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957,  0.8034, -0.6216]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p)
    >>> samples
    tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32
    >>> probs = torch.softmax(logits, dim=-1)
    >>> probs
    tensor([[0.4788, 0.3085, 0.1716, 0.0085, 0.0327],
        [0.2358, 0.1787, 0.4307, 0.1145, 0.0404],
        [0.1358, 0.2831, 0.1764, 0.2318, 0.1729],
        [0.3613, 0.1122, 0.1029, 0.3415, 0.0821]], device='cuda:0')
    >>> samples
    tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_k_mask_logits
    top_p_sampling_from_probs
    """
    if filter_apply_order == "top_k_first":
        masked_logits = top_k_mask_logits(logits, top_k)
        probs = torch.softmax(masked_logits, dim=-1)
        return top_p_sampling_from_probs(
            probs,
            top_p,
            indices,
            deterministic,
            check_nan=check_nan,
            generator=generator,
        )
    elif filter_apply_order == "joint":
        probs = torch.softmax(logits, dim=-1)
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return get_sampling_module().top_k_top_p_sampling_from_probs(
            probs,
            indices,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
            generator,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")
