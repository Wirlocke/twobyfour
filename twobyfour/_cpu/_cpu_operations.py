# Copyright (c) 2023 Chaoyang Wang
# 
# This source code contains snippets of work covered by MIT license.
# See LICENSE-dqtorch for the full license text

import torch

from . import rotation_conversions


def _quaternion_conjugate_pytorch(q: torch.Tensor) -> torch.Tensor:
    '''
        https://mathworld.wolfram.com/QuaternionConjugate.html
        when q is unit quaternion, inv(q) = conjugate(q)
    '''
    return torch.cat((q[..., 0:1], -q[..., 1:]), -1)


def _quaternion_4D_mul_3D(a: torch.Tensor, b_xyz: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bx, by, bz = torch.unbind(b_xyz, -1)
    ow = - ax * bx - ay * by - az * bz
    ox = aw * bx + ay * bz - az * by
    oy = aw * by - ax * bz + az * bx
    oz = aw * bz + ax * by - ay * bx
    return torch.stack((ow, ox, oy, oz), -1)


def _quaternion_3D_mul_4D(a_xyz: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az = torch.unbind(a_xyz, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = - ax * bx - ay * by - az * bz
    ox = ax * bw + ay * bz - az * by
    oy = - ax * bz + ay * bw + az * bx
    oz = ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def _quaternion_mul_pytorch(a: torch.Tensor, b: torch.Tensor):
    '''
        native pytorch implementation, only used as a baseline.
    '''
    if a.shape[-1] == 4 and b.shape[-1] == 4:
        return rotation_conversions.quaternion_raw_multiply(a, b)
    elif a.shape[-1] == 3 and b.shape[-1] == 4:
        return _quaternion_3D_mul_4D(a, b)
    elif a.shape[-1] == 4 and b.shape[-1] == 3:
        return _quaternion_4D_mul_3D(a, b)
    else:
        raise ValueError(f"Invalid input shapes.")
