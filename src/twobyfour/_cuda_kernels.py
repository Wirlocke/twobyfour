# Copyright (c) 2025 Marisha Norcross
#
# This source code is covered by MIT license
# See LICENSE for the full license text.


from typing import cast, Callable

import torch
from torch import Tensor

from . import _C  # type: ignore
from .typing import Quaternion
from . import _native_functions as native

tupleTensor2 = tuple[Tensor, Tensor]


# =============================================
# Quaternion Multiply
# =============================================


QUATERNION_MULTIPLY = "twobyfour::quaternion_multiply"


@torch.library.register_fake(QUATERNION_MULTIPLY)
def _(left: Tensor, right: Tensor) -> Tensor:
    torch._check(left.shape == right.shape)
    torch._check(left.is_floating_point() or left.is_complex())
    torch._check(right.is_floating_point() or right.is_complex())
    torch._check(left.dtype == right.dtype)
    torch._check(left.device == right.device)
    return torch.empty_like(left)


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    left = left.contiguous()
    right = right.contiguous()
    _q_m = cast(Callable[[Tensor, Tensor], Tensor],
                torch.ops.twobyfour.quaternion_multiply)
    return _q_m(left, right)


def _quat_mul_backward(ctx, grad: Tensor):
    leftconj, rightconj = cast(tupleTensor2, ctx.saved_tensors)

    grad_left, grad_right = None, None
    if ctx.needs_input_grad[0]:
        grad_left = _quaternion_multiply(grad, rightconj)
    if ctx.needs_input_grad[1]:
        grad_right = _quaternion_multiply(leftconj, grad)
    return grad_left, grad_right


def _quat_mul_setup_context(ctx, inputs: tupleTensor2, output):
    left, right = inputs

    saved_leftconj, saved_rightconj = None, None
    if ctx.needs_input_grad[0]:
        saved_rightconj = native._quaternion_conjugate(right)
    if ctx.needs_input_grad[1]:
        saved_leftconj = native._quaternion_conjugate(left)

    ctx.save_for_backward(saved_leftconj, saved_rightconj)


torch.library.register_autograd(
    QUATERNION_MULTIPLY, _quat_mul_backward, setup_context=_quat_mul_setup_context)


def quat_mul(left: Quaternion, right: Quaternion) -> Quaternion:
    leftbc, rightbc = cast(tupleTensor2,
                           torch.broadcast_tensors(left, right))

    output = _quaternion_multiply(leftbc, rightbc)
    return Quaternion(output)


# =============================================
# Quaternion Apply
# =============================================


QUATERNION_APPLY = "twobyfour::quaternion_apply"


@torch.library.register_fake(QUATERNION_APPLY)
def _(quat: Tensor, point: Tensor):
    torch._check(quat.shape == point.shape)
    torch._check(quat.is_floating_point() or quat.is_complex())
    torch._check(point.is_floating_point() or point.is_complex())
    torch._check(quat.dtype == point.dtype)
    torch._check(quat.device == point.device)
    return torch.empty_like(point)


def _quaternion_apply(quat: Tensor, point: Tensor) -> Tensor:
    quat = quat.contiguous()
    point = point.contiguous()
    _q_a = cast(Callable[[Tensor, Tensor], Tensor],
                torch.ops.twobyfour.quaternion_apply)
    return _q_a(quat, point)


def _quat_apply_backward(ctx, grad: Tensor):
    quat, point = cast(tupleTensor2, ctx.saved_tensors)
    quatconj = native._quaternion_conjugate(quat)
    pointconj = native._quaternion_conjugate(point)

    grad_quat, grad_point = None, None
    if ctx.needs_input_grad[0]:
        grad_quat = (_quaternion_multiply(_quaternion_multiply(grad, quat), pointconj) +
                     native._quaternion_conjugate(_quaternion_multiply(_quaternion_multiply(pointconj, quatconj), grad)))
    if ctx.needs_input_grad[1]:
        grad_point = _quaternion_apply(quatconj, grad)
    return grad_quat, grad_point


def _quat_apply_setup_context(ctx, inputs: tupleTensor2, output):
    quat, point = inputs

    saved_quat, saved_point = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        saved_quat = quat
        saved_point = point

    ctx.save_for_backward(saved_quat, saved_point)


torch.library.register_autograd(
    QUATERNION_APPLY, _quat_apply_backward, setup_context=_quat_apply_setup_context)


def quat_apply(quat: Quaternion, point: Quaternion) -> Quaternion:
    quatbc, pointbc = cast(tupleTensor2,
                           torch.broadcast_tensors(quat, point))

    output = _quaternion_apply(quatbc, pointbc)
    return Quaternion(output)
