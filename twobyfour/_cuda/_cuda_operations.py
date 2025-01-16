# Copyright (c) 2025 Marisha Norcross
# Copyright (c) 2023 Chaoyang Wang
#
# This source code contains modifications of work covered by MIT license.
# See LICENSE and LICENSE-dqtorch for the full license text.

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.amp.autocast_mode import custom_bwd, custom_fwd

from ._cuda_kernels import kernels


def _validate_broadcast_dimensions(B: int, B1: int, B2: int, D1: int, D2: int):
    if not (B1 == B2) and not (B1 == 1 or B2 == 1):
        raise ValueError(
            f"Batch dimensions must match or one must be scalar. Got B={B}, B1={B1}, B2={B2}"  # nopep8
        )

    if not (D1 == 3 or D1 == 4):
        raise ValueError(
            f"Tensor_1[-1] must be either 3 or 4 dimensions. Got D1={D1}"
        )

    if not (D2 == 3 or D2 == 4):
        raise ValueError(
            f"Tensor_2[-1] must be either 3 or 4 dimensions. Got D2={D2}"
        )


def _get_broadcast_meta_data(inputs_1: torch.Tensor, inputs_2: torch.Tensor):
    B1 = inputs_1.shape[0]  # batch size, coord dim
    B2 = inputs_2.shape[0]
    B = max(B1, B2)
    D1 = inputs_1.shape[-1]
    D2 = inputs_2.shape[-1]
    _validate_broadcast_dimensions(B, B1, B2, D1, D2)
    P1 = D1 == 3
    P2 = D2 == 3
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

        # if ctx.calc_grad_inputs:
        grad, *_ = grad_outputs

        inputs_1, inputs_2 = ctx.saved_tensors

        gi = _quaternion_mul_backward(grad, inputs_1, inputs_2)
        assert gi is tuple[torch.Tensor, torch.Tensor]
        grad_inputs_1, grad_inputs_2 = gi

        return grad_inputs_1, grad_inputs_2


quaternion_mul = _Quaternion_mul.apply


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
