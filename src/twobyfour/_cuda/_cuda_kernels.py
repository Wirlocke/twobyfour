# Copyright (c) 2025 Marisha Norcross
#
# This source code is covered by MIT license
# See LICENSE for the full license text.


from typing import cast, Callable

import torch
from torch import Tensor

from .. import _C  # type: ignore
from ..typing import Quaternion
from .. import operators as ops

tupleTensor = tuple[Tensor, Tensor]


@torch.library.register_fake("twobyfour::quaternion_multiply")
def _(left: Tensor, right: Tensor) -> Tensor:
    torch._check(left.dtype == torch.float)
    torch._check(left.dtype == torch.float)
    torch._check(left.device == right.device)

    return torch.empty_like(left)


quaternion_multiply = cast(Callable[[Tensor, Tensor], Tensor],
                           torch.ops.twobyfour.quaternion_multiply)


def _quat_mul_backward(ctx, grad: Tensor):
    left, right = cast(tupleTensor, ctx.saved_tensors)
    left[..., 1:] *= -1
    right[..., 1:] *= -1

    grad_left, grad_right = None, None
    if ctx.needs_input_grad[0]:
        grad_left = quaternion_multiply(grad.contiguous(), right.contiguous())
    if ctx.needs_input_grad[1]:
        grad_right = quaternion_multiply(left.contiguous(), grad.contiguous())
    return grad_left, grad_right, None


def _quat_mul_setup_context(ctx, inputs: tupleTensor, output):
    left, right = inputs

    saved_left, saved_right = None, None
    if ctx.needs_input_grad[0]:
        saved_left = left
    if ctx.needs_input_grad[1]:
        saved_right = right
    ctx.save_for_backward(saved_left, saved_right)


torch.library.register_autograd("twobyfour::quaternion_multiply",
                                _quat_mul_backward, setup_context=_quat_mul_setup_context)


def quat_mul(left: Quaternion, right: Quaternion) -> Quaternion:
    leftbc, rightbc = cast(tuple[Tensor, Tensor],
                           torch.broadcast_tensors(left, right))

    output = quaternion_multiply(leftbc.contiguous(), rightbc.contiguous())
    return Quaternion(output)
