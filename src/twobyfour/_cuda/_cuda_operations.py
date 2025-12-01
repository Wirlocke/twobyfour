from torch import Tensor
from torch.autograd import Function
from torch.amp.autocast_mode import custom_bwd, custom_fwd

from . import _cuda_kernels as kernel

CUDA = 'cuda'


class _quaternion_conjugate(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, inputs: Tensor):
        return kernel.quat_conj(inputs.contiguous())

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Tensor):
        return quat_conj(grad_outputs[0])


quat_conj = _quaternion_conjugate.apply


class _quaternion_multiply(Function):
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


quat_mul = _quaternion_multiply.apply


class _quaternion_squares(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, inputs: Tensor):
        ctx.save_for_backward(inputs)
        return kernel.quat_squares(inputs.contiguous())

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Tensor):
        grad_output = grad_outputs[0]
        inputs, = ctx.saved_tensors

        grad_out = 2 * inputs * grad_output
        return grad_out


quat_squares = _quaternion_squares.apply
