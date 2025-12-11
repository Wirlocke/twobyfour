from typing import cast, Callable

from torch import Tensor
from torch.autograd import Function
from torch.amp.autocast_mode import custom_bwd, custom_fwd

from ..typing import Quaternion
from . import _cuda_kernels as kernel


CUDA = 'cuda'


class _quaternion_multiply(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, in_left: Quaternion, in_right: Quaternion):
        ctx.save_for_backward(in_left, in_right)
        return kernel.quat_mul(in_left, in_right)

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Quaternion):
        grad_output = grad_outputs[0]
        saved_tensors: tuple[Quaternion, Quaternion] = ctx.saved_tensors
        in_left, in_right = saved_tensors

        in_left[..., 1:] *= -1
        in_right[..., 1:] *= -1

        left_grad = quat_mul(grad_output, in_right)
        right_grad = quat_mul(in_left, grad_output)
        return left_grad, right_grad


quat_mul = cast(Callable[[Quaternion, Quaternion], Quaternion],
                _quaternion_multiply.apply)
