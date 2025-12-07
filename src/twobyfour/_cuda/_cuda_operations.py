from typing import Callable
from typing import cast

from torch import Tensor
from torch.autograd import Function
from torch.amp.autocast_mode import custom_bwd, custom_fwd

from . import _cuda_kernels as kernel
from ..typing import Quaternion

CUDA = 'cuda'


class _quaternion_squared_sum(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, inputs: Quaternion):
        ctx.save_for_backward(inputs)
        return kernel.quat_sqsum(inputs)

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Tensor):
        grad_output = grad_outputs[0]
        inputs, = ctx.saved_tensors

        return 2 * inputs * grad_output


quat_sqsum = cast(Callable[[Quaternion], Tensor],
                  _quaternion_squared_sum.apply)


class _quaternion_magnitude(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, inputs: Quaternion):
        output = kernel.quat_mag(inputs)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Tensor):
        grad_output = grad_outputs[0]
        inputs, output = ctx.saved_tensors

        return (inputs / output) * grad_output


quat_mag = cast(Callable[[Quaternion], Tensor],
                _quaternion_magnitude.apply)


class _quaternion_normalize(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, inputs: Quaternion):
        output = kernel.quat_norm(inputs)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Quaternion):
        grad_output = grad_outputs[0]
        inputs, output = ctx.saved_tensors

        return (grad_output - (output * quat_dot(output, grad_output))) / quat_mag(inputs)


quat_norm = cast(Callable[[Quaternion], Quaternion],
                 _quaternion_normalize.apply)


class _quaternion_conjugate(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, inputs: Quaternion):
        return kernel.quat_conj(inputs)

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Quaternion):
        return quat_conj(grad_outputs[0])


quat_conj = cast(Callable[[Quaternion], Quaternion],
                 _quaternion_conjugate.apply)


class _quaternion_inverse(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, inputs: Quaternion):
        output = kernel.quat_inv(inputs)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Quaternion):
        grad_output = grad_outputs[0]
        inputs, output = ctx.saved_tensors

        return (quat_conj(grad_output) - (2 * inputs * quat_dot(output, grad_output))) / quat_sqsum(inputs)


quat_inv = cast(Callable[[Quaternion], Quaternion],
                _quaternion_inverse.apply)


class _quaternion_dot_product(Function):
    @staticmethod
    @custom_fwd(device_type=CUDA)
    def forward(ctx, input1: Quaternion, input2: Quaternion):
        ctx.save_for_backward(input1, input2)
        return kernel.quat_dot(input1, input2)

    @staticmethod
    @custom_bwd(device_type=CUDA)
    def backward(ctx, *grad_outputs: Tensor):
        grad_output = grad_outputs[0]
        input1, input2 = ctx.saved_tensors

        grad_input1 = input2 * grad_output
        grad_input2 = input1 * grad_output
        return grad_input1, grad_input2


quat_dot = cast(Callable[[Quaternion, Quaternion], Tensor],
                _quaternion_dot_product.apply)


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

        left_grad = quat_mul(grad_output, quat_conj(in_right))
        right_grad = quat_mul(quat_conj(in_left), grad_output)
        return left_grad, right_grad


quat_mul = cast(Callable[[Quaternion, Quaternion], Quaternion],
                _quaternion_multiply.apply)
