# Copyright (c) 2025 Marisha Norcross
#
# This source code is covered by MIT license
# See LICENSE for the full license text.


from typing import cast, Callable

import torch
from torch import Tensor

from . import _C  # type: ignore
from .typing import Quaternion

tupleTensor2 = tuple[Tensor, Tensor]


# =============================================
# Quaternion Multiply
# =============================================


QUATERNION_MULTIPLY = "twobyfour::quaternion_multiply"


@torch.library.register_fake(QUATERNION_MULTIPLY)
def _(left: Tensor, right: Tensor) -> Tensor:
    torch._check(left.shape == right.shape)
    torch._check(left.dtype == torch.float)
    torch._check(left.dtype == torch.float)
    torch._check(left.device == right.device)
    return torch.empty_like(left)


_quaternion_multiply = cast(Callable[[Tensor, Tensor], Tensor],
                            torch.ops.twobyfour.quaternion_multiply)


def _quat_mul_backward(ctx, grad: Tensor):
    grad = grad.contiguous()
    leftconj, rightconj = cast(tupleTensor2, ctx.saved_tensors)
    leftconj[..., 1:] *= -1
    rightconj[..., 1:] *= -1

    grad_left, grad_right = None, None
    if ctx.needs_input_grad[0]:
        grad_left = _quaternion_multiply(grad, rightconj)
    if ctx.needs_input_grad[1]:
        grad_right = _quaternion_multiply(leftconj, grad)
    return grad_left, grad_right


def _quat_mul_setup_context(ctx, inputs: tupleTensor2, output):
    left, right = inputs

    saved_left, saved_right = None, None
    if ctx.needs_input_grad[0]:
        saved_left = left.contiguous()
    if ctx.needs_input_grad[1]:
        saved_right = right.contiguous()
    ctx.save_for_backward(saved_left, saved_right)


torch.library.register_autograd(
    QUATERNION_MULTIPLY, _quat_mul_backward, setup_context=_quat_mul_setup_context)


def quat_mul(left: Quaternion, right: Quaternion) -> Quaternion:
    leftbc, rightbc = cast(tupleTensor2,
                           torch.broadcast_tensors(left, right))

    output = _quaternion_multiply(leftbc.contiguous(), rightbc.contiguous())
    return Quaternion(output)


# =============================================
# Quaternion Apply
# =============================================


QUATERNION_APPLY = "twobyfour::quaternion_apply"


@torch.library.register_fake(QUATERNION_APPLY)
def _(quat: Tensor, point: Tensor):
    torch._check(quat.shape == point.shape)
    torch._check(quat.dtype == torch.float)
    torch._check(point.dtype == torch.float)
    torch._check(quat.device == point.device)
    return torch.empty_like(point)


_quaternion_apply = cast(Callable[[Tensor, Tensor], Tensor],
                         torch.ops.twobyfour.quaternion_apply)


def _quat_apply_backward(ctx, grad: Tensor):
    grad = grad.contiguous()
    quat, point = cast(tupleTensor2, ctx.saved_tensors)
    quat_conj = quat.clone()
    quat_conj[..., 1:] *= -1

    grad_quat, grad_point = None, None
    if ctx.needs_input_grad[0]:
        grad_quat = 2 * (_quaternion_multiply(grad, _quaternion_multiply(point, quat_conj)) +
                         _quaternion_multiply(quat_conj, _quaternion_multiply(grad, point)))
    if ctx.needs_input_grad[1]:
        grad_point = _quaternion_apply(quat, grad)
    return grad_quat, grad_point


def _quat_apply_setup_context(ctx, inputs: tupleTensor2, output):
    quat, point = inputs

    saved_quat, saved_point = None, None
    if ctx.needs_input_grad[0]:
        saved_quat = quat.contiguous()
    if ctx.needs_input_grad[1]:
        saved_point = point.contiguous()
    ctx.save_for_backward(saved_quat, saved_point)


torch.library.register_autograd(
    QUATERNION_APPLY, _quat_apply_backward, setup_context=_quat_apply_setup_context)


def quat_apply(quat: Quaternion, point: Quaternion) -> Quaternion:
    quatbc, pointbc = cast(tupleTensor2,
                           torch.broadcast_tensors(quat, point))

    output = _quaternion_apply(quatbc.contiguous(), pointbc.contiguous())
    return Quaternion(output)
