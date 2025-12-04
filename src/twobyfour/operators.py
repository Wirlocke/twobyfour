# Copyright (c) 2025 Marisha Norcross
# Copyright (c) 2023 Chaoyang Wang
#
# This source code contains modifications of work covered by MIT license.
# See LICENSE and LICENSE-dqtorch for the full license text.

from typing import Tuple, cast

import torch

from ._cpu import _cpu_operations as cpu
from ._cuda import _cuda_operations as cuda

Quaternion = torch.Tensor
DualQuaternions = Tuple[Quaternion, Quaternion]
QuaternionTranslation = Tuple[Quaternion, torch.Tensor]


# =============================================
# Helper Functions
# =============================================


def tensor_cast(t, shape: torch.Size) -> torch.Tensor:
    return cast(torch.Tensor, t).view(shape)


def tcast(t, shape: torch.Size) -> torch.Tensor:
    return tensor_cast(t, shape)


def quat_flatten(q: Quaternion) -> torch.Tensor:
    return q.contiguous().view(-1, 4)


def qflat(q: Quaternion) -> torch.Tensor:
    return quat_flatten(q)


# =============================================
# Single Input Quaternion Operations
# =============================================


def quaternion_squares(q: Quaternion) -> torch.Tensor:
    if q.is_cuda:
        out_shape = torch.Size(q.shape[:-1] + (1,))
        return tcast(cuda.quat_sqsum(qflat(q)), out_shape)
    else:
        return q.pow(2).sum(-1, keepdim=True)


def squardsumq(q: Quaternion) -> torch.Tensor:
    return quaternion_squares(q)


def quaternion_magnitude(q: Quaternion) -> torch.Tensor:
    if q.is_cuda:
        out_shape = torch.Size(q.shape[:-1] + (1,))
        return tcast(cuda.quat_mag(qflat(q)), out_shape)
    else:
        return quaternion_squares(q).sqrt()


def magq(q: Quaternion) -> torch.Tensor:
    return quaternion_magnitude(q)


def quaternion_normalize(q: Quaternion) -> Quaternion:
    if q.is_cuda:
        out_shape = q.shape
        return tcast(cuda.quat_norm(qflat(q)), out_shape)
    else:
        return q / quaternion_magnitude(q)


def normq(q: Quaternion) -> Quaternion:
    return quaternion_normalize(q)


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    if q.is_cuda:
        out_shape = q.shape
        return tcast(cuda.quat_conj(qflat(q)), out_shape)
    else:
        return cpu._quaternion_conjugate_pytorch(q)


def conjq(q: Quaternion) -> Quaternion:
    return quaternion_conjugate(q)


def quaternion_inverse(q: Quaternion) -> Quaternion:
    if q.is_cuda:
        out_shape = q.shape
        return tcast(cuda.quat_inv(qflat(q)), out_shape)
    else:
        return quaternion_conjugate(q) / quaternion_squares(q)


def invq(q: Quaternion) -> Quaternion:
    return quaternion_inverse(q)


# =============================================
# Multi Input Quaternion Operations
# =============================================

def quaternion_dot_product(a: Quaternion, b: Quaternion) -> torch.Tensor:
    if a.is_cuda:
        out_shape = torch.Size(a.shape[:-1] + (1,))
        return tcast(cuda.quat_dot(qflat(a), qflat(b)), out_shape)
    else:
        return (a * b).sum(-1, keepdim=True)


def dotq(a: Quaternion, b: Quaternion) -> torch.Tensor:
    return quaternion_dot_product(a, b)


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    if left.is_cuda:
        out_shape = left.shape
        return tcast(cuda.quat_mul(qflat(left), qflat(right)), out_shape)
    else:
        return cpu._quaternion_mul_pytorch(left, right)


def mulq(left: Quaternion, right: Quaternion) -> Quaternion:
    return quaternion_multiply(left, right)


def quaternion_apply(quaternion: Quaternion, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    out = quaternion_multiply(
        quaternion_multiply(quaternion, point),
        quaternion_conjugate(quaternion)
    )
    return out[..., 1:].contiguous()


def applyq(quaternion: Quaternion, point: torch.Tensor) -> torch.Tensor:
    return quaternion_apply(quaternion, point)


# =============================================
# Quaternion Translations
# =============================================


def quaternion_translation_apply(q: Quaternion, t: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    p = quaternion_apply(q, point)
    return p + t


def quaternion_translation_compose(qt1: QuaternionTranslation, qt2: QuaternionTranslation) -> QuaternionTranslation:
    qr = quaternion_multiply(qt1[0], qt2[0])
    t = quaternion_apply(qt1[0], qt2[1]) + qt1[1]
    return (qr, t)


def quaternion_translation_inverse(q: Quaternion, t: torch.Tensor) -> QuaternionTranslation:
    q_inv = quaternion_conjugate(q)
    t_inv = quaternion_apply(q_inv, -t)
    return q_inv, t_inv

# =============================================
# Conversions
# =============================================


def quaternion_translation_to_dual_quaternion(
        q: torch.Tensor, t: torch.Tensor) -> DualQuaternions:
    '''
    https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    '''
    q_d = 0.5 * quaternion_multiply(t, q)
    return (q, q_d)


def dual_quaternion_to_quaternion_translation(dq: DualQuaternions) -> DualQuaternions:
    q_r, q_d = dq
    t = 2*quaternion_multiply(q_d, quaternion_conjugate(q_r))[..., 1:]

    return q_r, t


# =============================================
# Single Input Dual Quaternion Operations
# =============================================


def dual_quaternion_inner_dot_product(dq: DualQuaternions) -> torch.Tensor:
    q_r, q_d = dq
    return quaternion_dot_product(q_r, q_d)


def dual_quaternion_q_conjugate(dq: DualQuaternions) -> DualQuaternions:
    q_r = quaternion_conjugate(dq[0])
    q_d = quaternion_conjugate(dq[1])
    return (q_r, q_d)


def dual_quaternion_d_conjugate(dq: DualQuaternions) -> DualQuaternions:
    return (dq[0], -dq[1])


def dual_quaternion_3rd_conjugate(dq: DualQuaternions) -> DualQuaternions:
    return dual_quaternion_d_conjugate(dual_quaternion_q_conjugate(dq))


def dual_quaternion_unit_inverse(dq: DualQuaternions) -> DualQuaternions:
    return dual_quaternion_q_conjugate(dq)


def dual_quaternion_inverse(dq: DualQuaternions) -> DualQuaternions:
    real, _ = dq

    sq_sum_real = quaternion_squares(real)
    inner_dot = dual_quaternion_inner_dot_product(dq)
    conj_real, conj_dual = dual_quaternion_q_conjugate(dq)

    inv_real = conj_real / sq_sum_real
    inv_dual = (conj_dual - (2 * inner_dot * inv_real)) / sq_sum_real

    return (inv_real, inv_dual)


# =============================================
# Multi Input Dual Quaternion Operations
# =============================================


def dual_quaternion_multiply(left: DualQuaternions, right: DualQuaternions) -> DualQuaternions:
    real_left, dual_left = left
    real_right, dual_right = right

    real_out = quaternion_multiply(real_left, real_right)

    dual_out = quaternion_multiply(real_left, dual_right) + \
        quaternion_multiply(dual_left, real_right)

    return (real_out, dual_out)


def dual_quaternion_apply(dq: DualQuaternions, point: torch.Tensor) -> torch.Tensor:
    """
    assuming the input dual quaternion is normalized.
    """
    q, t = dual_quaternion_to_quaternion_translation(dq)
    return quaternion_translation_apply(q, t, point)


def dual_quaternion_rectify(dq: DualQuaternions) -> DualQuaternions:
    """
    input: (unit quaternion, 4D vector w') -> dual quaternion, which satisfies (r, 0.5 * t r)
    solve: min | q - w' | s.t. w^T r = 0
    """
    q_r, q_d = dq
    q_d = q_d - quaternion_dot_product(q_r, q_d) * q_r

    return (q_r, q_d)
