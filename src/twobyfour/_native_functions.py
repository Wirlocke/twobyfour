import torch
from torch import Tensor


# =============================================
# Single Input Quaternion Operations
# =============================================


def _quaternion_squares(q: Tensor) -> Tensor:
    return q.square().sum(-1, keepdim=True)


def _quaternion_magnitude(q: Tensor) -> Tensor:
    return q.norm(dim=-1, keepdim=True)


def _quaternion_normalize(q: Tensor) -> Tensor:
    return q / _quaternion_magnitude(q)


def _quaternion_conjugate(q: Tensor) -> Tensor:
    result = q.clone()
    result[..., 1:] *= -1
    return result


def _quaternion_inverse(q: Tensor) -> Tensor:
    return _quaternion_conjugate(q) / _quaternion_squares(q)


# =============================================
# Multi Input Quaternion Operations
# =============================================


def _quaternion_dot_product(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(-1, keepdim=True)


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    la, lb, lc, ld = left.unbind(-1)
    ra, rb, rc, rd = right.unbind(-1)
    result = torch.stack((
        la * ra - lb * rb - lc * rc - ld * rd,
        la * rb + lb * ra + lc * rd - ld * rc,
        la * rc - lb * rd + lc * ra + ld * rb,
        la * rd + lb * rc - lc * rb + ld * ra
    ), dim=-1)
    return result


def _quaternion_apply(quaternion: Tensor, point: Tensor) -> Tensor:
    return _quaternion_multiply(
        _quaternion_multiply(quaternion, point),
        _quaternion_conjugate(quaternion)
    )
