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
# Basic Quaternion Operations
# =============================================


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    # out_shape = q.shape
    # return _quaternion_conjugate_cuda(q.contiguous().view(-1,4)).view(out_shape)
    if q.is_cuda:
        out_shape = q.shape
        return cast(torch.Tensor,
                    cuda.quat_conj(
                        q.contiguous().view(-1, 4))
                    ).view(out_shape)
    else:
        return cpu._quaternion_conjugate_pytorch(q)


def quaternion_mul(a: Quaternion, b: Quaternion) -> Quaternion:
    if a.is_cuda:
        ouput_shape = list(a.shape[:-1]) + [4]
        return cast(torch.Tensor,
                    cuda.quat_mul(
                        a.view(-1, a.shape[-1]),
                        b.view(-1, b.shape[-1]))
                    ).view(ouput_shape)
    else:
        return cpu._quaternion_mul_pytorch(a, b)


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
    out = quaternion_mul(
        quaternion_mul(quaternion, point),
        quaternion_conjugate(quaternion)
    )
    return out[..., 1:].contiguous()

# =============================================
# Quaternion Translations
# =============================================


def quaternion_translation_apply(q: Quaternion, t: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    p = quaternion_apply(q, point)
    return p + t


def quaternion_translation_compose(qt1: QuaternionTranslation, qt2: QuaternionTranslation) -> QuaternionTranslation:
    qr = quaternion_mul(qt1[0], qt2[0])
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
    q_d = 0.5 * quaternion_mul(t, q)
    return (q, q_d)


def dual_quaternion_to_quaternion_translation(dq: DualQuaternions) -> DualQuaternions:
    q_r, q_d = dq
    t = 2*quaternion_mul(q_d, quaternion_conjugate(q_r))[..., 1:]

    return q_r, t

# =============================================
# Dual Quaternion Conjugates
# =============================================


def dual_quaternion_q_conjugate(dq: DualQuaternions) -> DualQuaternions:
    r = quaternion_conjugate(dq[0])
    d = quaternion_conjugate(dq[1])
    return (r, d)


dual_quaternion_inverse = dual_quaternion_q_conjugate


def dual_quaternion_d_conjugate(dq: DualQuaternions) -> DualQuaternions:
    return (dq[0], -dq[1])


def dual_quaternion_3rd_conjugate(dq: DualQuaternions) -> DualQuaternions:
    return dual_quaternion_d_conjugate(dual_quaternion_q_conjugate(dq))

# =============================================
# Dual Quaternion Operations
# =============================================


def dual_quaternion_mul(dq1: DualQuaternions, dq2: DualQuaternions) -> DualQuaternions:
    q_r1, q_d1 = dq1
    q_r2, q_d2 = dq2

    r_r = quaternion_mul(q_r1, q_r2)
    r_d = quaternion_mul(q_r1, q_d2) + quaternion_mul(q_d1, q_r2)
    return (r_r, r_d)


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
    q_d = q_d - (q_r * q_d).sum(-1, keepdim=True) * q_r

    return (q_r, q_d)
