from torch import Tensor
from torch.autograd import Function
from torch.amp.autocast_mode import custom_bwd, custom_fwd

from . import _cuda_kernels as kernel

CUDA = 'cuda'

# Replace with inverse
class _Quaternion_conj(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, inputs: Tensor):
        return kernel.quat_conj(inputs.contiguous())

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Tensor):
        return quat_conj(grad_outputs[0])


quat_conj = _Quaternion_conj.apply


class _Quaternion_mul(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, in_left: Tensor, in_right: Tensor):
        in_left = in_left.contiguous()
        in_right = in_right.contiguous()
        ctx.save_for_backward(in_left, in_right)
        return kernel.quat_mul(in_left, in_right)

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Tensor):
        grad_output = grad_outputs[0]
        saved_tensors: tuple[Tensor, Tensor] = ctx.saved_tensors
        in_left, in_right = saved_tensors

        left_grad = quat_mul(grad_output, quat_conj(in_right))
        right_grad = quat_mul(quat_conj(in_left), grad_output)
        return left_grad, right_grad


quat_mul = _Quaternion_mul.apply
