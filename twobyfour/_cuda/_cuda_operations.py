# Copyright (c) 2025 Marisha Norcross
# Copyright (c) 2023 Chaoyang Wang
#
# This source code contains modifications of work covered by MIT license.
# See LICENSE and LICENSE-dqtorch for the full license text.

import sys

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.amp.autocast_mode import custom_bwd, custom_fwd

from ._cuda_kernels import kernels

EPS = sys.float_info.epsilon


def _validate_dimensions(D: int):
    if not (D == 3 or D == 4):
        raise ValueError(
            f"Tensor[-1] must be either 3 or 4 dimensions. Got D={D}"
        )


def _get_meta_data(inputs: torch.Tensor):
    B = inputs.shape[0]
    D = inputs.shape[-1]
    _validate_dimensions(D)
    P = D == 3
    return B, D, P


def _validate_broadcast_dimensions(B: int, B1: int, B2: int):
    if not (B1 == B2) and not (B1 == 1 or B2 == 1):
        raise ValueError(
            f"Batch dimensions must match or one must be scalar. Got B={B}, B1={B1}, B2={B2}"  # nopep8
        )


def _get_broadcast_meta_data(inputs_1: torch.Tensor, inputs_2: torch.Tensor):
    B1, D1, P1 = _get_meta_data(inputs_1)
    B2, D2, P2 = _get_meta_data(inputs_2)
    B = max(B1, B2)
    _validate_broadcast_dimensions(B, B1, B2)
    return B, B1, B2, D1, D2, P1, P2


class _Quaternion_mul_backward(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type='cuda')
    def forward(ctx, grad: torch.Tensor, inputs_1: torch.Tensor, inputs_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, B1, B2, D1, D2, P1, P2 = _get_broadcast_meta_data(
            inputs_1, inputs_2)

        dtype, device = inputs_1.dtype, inputs_1.device
        grad_inputs_1 = torch.empty(B1, D1, device=device, dtype=dtype)
        grad_inputs_2 = torch.empty(B2, D2, device=device, dtype=dtype)

        block_x = 256
        block = (block_x, 1, 1)

        grid_x = -((B*4) // -block_x)
        grid = (grid_x, 1, 1)

        kernels.kernel_quaternion_mul_backward(
            grad, B, P1, P2,
            inputs_1, inputs_2,
            grad_inputs_1, grad_inputs_2,
            block=block, grid=grid)

        ctx.save_for_backward(grad, inputs_1, inputs_2)

        return grad_inputs_1, grad_inputs_2

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_out_1, grad_out_2 = grad_outputs

        grad, inputs_1, inputs_2 = ctx.saved_tensors
        B, B1, B2, D1, D2, P1, P2 = _get_broadcast_meta_data(
            inputs_1, inputs_2)
        dtype, device = inputs_1.dtype, inputs_1.device

        grad_grad = torch.empty(B, 4, device=device, dtype=dtype)
        grad_grad_inputs_1 = torch.empty(B1, D1, device=device, dtype=dtype)
        grad_grad_inputs_2 = torch.empty(B2, D2, device=device, dtype=dtype)

        block_x = 256
        block = (block_x, 1, 1)

        grid_x = -((B*4) // -block_x)
        grid = (grid_x, 1, 1)

        kernels.kernel_quaternion_mul_backward_backward(
            grad_out_1, grad_out_2,
            B, P1, P2,
            grad, inputs_1, inputs_2,
            grad_grad, grad_grad_inputs_1, grad_grad_inputs_2,
            block=block, grid=grid)

        return grad_grad, grad_grad_inputs_1, grad_grad_inputs_2


_quaternion_mul_backward = _Quaternion_mul_backward.apply


class _Quaternion_mul(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type='cuda')
    def forward(ctx, inputs_1: torch.Tensor, inputs_2: torch.Tensor) -> torch.Tensor:
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float
        # calc_grad_inputs = inputs_1.requires_grad or inputs_2.requires_grad
        B, _, _, _, _, P1, P2 = _get_broadcast_meta_data(inputs_1, inputs_2)

        dtype = inputs_1.dtype
        device = inputs_1.device

        outputs = torch.empty(B, 4, dtype=dtype, device=device)

        block_x = 256
        block = (block_x, 1, 1)

        grid_x = -(B // -block_x)
        grid = (grid_x, 1, 1)

        kernels.kernel_quaternion_mul(
            inputs_1, inputs_2, outputs,
            B, P1, P2,
            block=block, grid=grid)

        ctx.save_for_backward(inputs_1, inputs_2)

        return outputs

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, *grad_outputs: torch.Tensor):
        # grad: [B, C * C]
        grad = grad_outputs[0]

        inputs_1, inputs_2 = ctx.saved_tensors

        gi = _quaternion_mul_backward(grad, inputs_1, inputs_2)
        assert type(gi) is tuple[torch.Tensor, torch.Tensor]
        grad_inputs_1, grad_inputs_2 = gi

        return grad_inputs_1, grad_inputs_2


quaternion_mul = _Quaternion_mul.apply


class _Quaternion_magnitude(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type='cuda')
    def forward(ctx, inputs: torch.Tensor):
        B, _, P = _get_meta_data(inputs)

        dtype = inputs.dtype
        device = inputs.device

        outputs = torch.empty(B, 1, dtype=dtype, device=device)

        block_x = 256
        block = (block_x, 1, 1)

        grid_x = -(B // -block_x)
        grid = (grid_x, 1, 1)

        kernels.kernel_quaternion_magnitude(
            inputs, B, P, outputs,
            block=block, grid=grid)

        ctx.save_for_backward(inputs, outputs.clone())

        return outputs

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, *grad_outputs: torch.Tensor):
        st = ctx.saved_tensors
        inputs, magnitude = st

        grad_out = grad_outputs[0]
        grad: torch.Tensor = inputs / (magnitude + EPS)
        return grad * grad_out


quaternion_magnitude = _Quaternion_magnitude.apply


class _Quaternion_conjugate(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type='cuda')
    def forward(ctx, inputs: torch.Tensor):
        B = inputs.shape[0]  # batch size, coord dim
        outputs = torch.empty_like(inputs)

        block_x = 256
        block = (block_x, 1, 1)

        grid_x = -((B*4) // -block_x)
        grid = (grid_x, 1, 1)

        kernels.kernel_quaternion_conjugate(
            inputs.contiguous(), B, outputs, block=block, grid=grid)
        return outputs

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, *grad_outputs: torch.Tensor):
        return _Quaternion_conjugate.apply(grad_outputs)


quaternion_conjugate = _Quaternion_conjugate.apply
