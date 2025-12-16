# Copyright (c) 2025 Marisha Norcross
# Copyright (c) 2023 Chaoyang Wang
#
# This source code contains modifications of work covered by MIT license.
# See LICENSE and LICENSE-dqtorch for the full license text.

from typing import Tuple

import torch
from torch import Tensor

from .typing import Quaternion
from . import _cuda_kernels as cuda


DualQuaternions = Tuple[Quaternion, Quaternion]
QuaternionTranslation = Tuple[Quaternion, Quaternion]


# =============================================
# Single Input Quaternion Operations
# =============================================


def quaternion_squares(q: Quaternion) -> Tensor:
    return q.square().sum(-1, keepdim=True)


def sqsumq(q: Quaternion) -> Tensor:
    return quaternion_squares(q)


def quaternion_magnitude(q: Quaternion) -> Tensor:
    return q.norm(dim=-1, keepdim=True)


def magq(q: Quaternion) -> Tensor:
    return quaternion_magnitude(q)


def quaternion_normalize(q: Quaternion) -> Quaternion:
    return q / quaternion_magnitude(q)


def normq(q: Quaternion) -> Quaternion:
    return quaternion_normalize(q)


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    result = q.clone()
    result[..., 1:] *= -1
    return result


def conjq(q: Quaternion) -> Quaternion:
    return quaternion_conjugate(q)


def quaternion_inverse(q: Quaternion) -> Quaternion:
    return quaternion_conjugate(q) / quaternion_squares(q)


def invq(q: Quaternion) -> Quaternion:
    return quaternion_inverse(q)


# =============================================
# Multi Input Quaternion Operations
# =============================================

def quaternion_dot_product(a: Quaternion, b: Quaternion) -> Tensor:
    return (a * b).sum(-1, keepdim=True)


def dotq(a: Quaternion, b: Quaternion) -> Tensor:
    return quaternion_dot_product(a, b)


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    if left.is_cuda:
        return cuda.quat_mul(left, right)
    else:
        la, lb, lc, ld = left.unbind(-1)
        ra, rb, rc, rd = right.unbind(-1)
        result = torch.stack((
            la * ra - lb * rb - lc * rc - ld * rd,
            la * rb + lb * ra + lc * rd - ld * rc,
            la * rc - lb * rd + lc * ra + ld * rb,
            la * rd + lb * rc - lc * rb + ld * ra
        ), dim=-1)
        return Quaternion(result)


def mulq(left: Quaternion, right: Quaternion) -> Quaternion:
    return quaternion_multiply(left, right)


def quaternion_apply(quaternion: Quaternion, point: Quaternion) -> Quaternion:
    if quaternion.is_cuda:
        return cuda.quat_apply(quaternion, point)
    else:
        return quaternion_multiply(
            quaternion_multiply(quaternion, point),
            quaternion_conjugate(quaternion)
        )


def applyq(quaternion: Quaternion, point: Quaternion) -> Quaternion:
    return quaternion_apply(quaternion, point)


# =============================================
# Quaternion Translations
# =============================================


def quaternion_translation_apply(q: Quaternion, t: Quaternion, point: Quaternion) -> Quaternion:
    p = quaternion_apply(q, point)
    return p + t


def quaternion_translation_compose(qt1: QuaternionTranslation, qt2: QuaternionTranslation) -> QuaternionTranslation:
    qr = quaternion_multiply(qt1[0], qt2[0])
    t = quaternion_apply(qt1[0], qt2[1]) + qt1[1]
    return (qr, t)


def quaternion_translation_inverse(q: Quaternion, t: Quaternion) -> QuaternionTranslation:
    q_inv = quaternion_conjugate(q)
    t_inv = quaternion_apply(q_inv, -t)
    return q_inv, t_inv

# =============================================
# Conversions
# =============================================


def quaternion_translation_to_dual_quaternion(q: Quaternion, t: Quaternion) -> DualQuaternions:
    '''
    https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    '''
    q_d = 0.5 * quaternion_multiply(t, q)
    return (q, q_d)


def dual_quaternion_to_quaternion_translation(dq: DualQuaternions) -> QuaternionTranslation:
    q_r, q_d = dq
    t = 2*quaternion_multiply(q_d, quaternion_conjugate(q_r))

    return q_r, t


# =============================================
# Single Input Dual Quaternion Operations
# =============================================


def dual_quaternion_inner_dot_product(dq: DualQuaternions) -> Tensor:
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

    dual_out = quaternion_multiply(real_left, dual_right) \
        + quaternion_multiply(dual_left, real_right)

    return (real_out, dual_out)


def dual_quaternion_apply(dq: DualQuaternions, point: Quaternion) -> Tensor:
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
