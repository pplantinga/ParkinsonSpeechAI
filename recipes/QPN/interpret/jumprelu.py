import torch
import torch.nn.functional as F


class JumpReLU(torch.nn.Module):
    """Activation function that zeros all values less than a jump_value.

    This is good for some security and SAE applications, as the module
    only outputs more confident predictions, see https://arxiv.org/abs/2407.14435v1
    This paper covers implementation details, but we use this repo as reference:

    https://github.com/saprmarks/dictionary_learning

    Arguments
    ---------
    input_size: int, optional
        Number of neurons in the input to establish per-input thresholds
    static_threshold: float, optional
        Static global threshold value, used instead of per-input thresholds
    initial_threshold: non-negative float, default 0.001
        Initial threshold value to apply across inputs
    grad_bandwidth: float, default 0.001
        The width of the gradient around the threshold.
    """

    def __init__(
        self,
        input_size=None,
        static_threshold=None,
        initial_threshold=0.001,
        grad_bandwidth=0.001,
    ):
        super().__init__()

        if (input_size is None) is (static_threshold is None):
            raise ValueError(
                "Must specify exactly one of input_size and static_threshold"
            )

        self.bandwidth = grad_bandwidth
        if input_size is not None:
            initial_threshold = torch.full((input_size,), initial_threshold)
            self.threshold = torch.nn.Parameter(initial_threshold)
        else:
            self.threshold = torch.tensor(static_threshold)

    def forward(self, x, sparse_loss=False):
        """Returns x with all values < threshold zeroed out.

        Arguments
        ---------
        x: torch.Tensor
            Tensor on which to apply JumpReLU activations
        sparse_loss: bool
            Whether to additionally return a sparsity criterion (l0)

        Returns
        -------
        x: torch.Tensor
            input with JumpReLU activations applied
        l0_loss: torch.Tensor (conditional)
            Returned if `sparse_loss` is `True`
        """
        x = JumpReLUFunction.apply(F.relu(x), self.threshold, self.bandwidth)

        if sparse_loss:
            x = (x, StepFunction.apply(x, self.threshold, self.bandwidth))

        return x


class RectangleFunction(torch.autograd.Function):
    """Rectangle / delta function used in JumpReLU and StepFunction

    See https://arxiv.org/abs/2407.14435v1 for implementation details.

    Implementation from: saprmarks/dictionary_learning github
    Implementation file: dictionary_learning/trainers/jumprelu.py
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(torch.autograd.Function):
    """Companion to JumpReLU module with pseudo-differentiation code.

    Uses the straight-through-estimator (STE) trick in a small neighborhood
    of the threshold value, defined here as JUMP_RELU_BANDWIDTH.

    See https://arxiv.org/abs/2407.14435v1 for implementation details.

    Implementation from: saprmarks/dictionary_learning github
    Implementation file: dictionary_learning/trainers/jumprelu.py
    """

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(torch.autograd.Function):
    """Heaviside step function with custom backwards.

    Not intended for ordinary activations, used for l0-loss computation.

    Uses the straight-through-estimator (STE) trick in a small neighborhood
    of the threshold value, the "bandwidth".

    See https://arxiv.org/abs/2407.14435v1 for implementation details.

    Implementation from: saprmarks/dictionary_learning github
    Implementation file: dictionary_learning/trainers/jumprelu.py
    """

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth
