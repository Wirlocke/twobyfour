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


def _quaternion_cross_product(a: Tensor, b: Tensor) -> Tensor:
    return torch.cross(a, b, dim=-1)


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


def _quaternion_unit_apply(quaternion: Tensor, point: Tensor) -> Tensor:
    quat_w = quaternion[..., 0:1]
    quat_v = quaternion[..., 1:]
    point_v = point[..., 1:]

    cross_prod = _quaternion_cross_product(quat_v, point_v)
    double_cross = _quaternion_cross_product(quat_v, cross_prod)

    result_v = point_v + (2 * quat_w * cross_prod) + (2 * double_cross)

    return torch.cat([point[..., 0:1], result_v], dim=-1)
