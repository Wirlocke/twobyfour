# Copyright (c) 2025 Marisha Norcross
#
# This source code is covered by MIT license
# See LICENSE for the full license text.

from pathlib import Path
from typing import cast

import cutex
import torch
from torch import Tensor

from twobyfour.typing import Quaternion

DIR = Path(__file__).parent / "kernels.cu"

with open(DIR, 'r') as file:
    KERNELS = file.read()

kernels = cutex.SourceModule(KERNELS, float_bits=32, boundscheck=False)

TUPLE_XYZ = tuple[int, int, int]


def cast_quaternion(input) -> Quaternion:
    return cast(Quaternion, input)


def block_grid_dim(input: Tensor, block_x=256, block_y=1, block_z=4) -> tuple[TUPLE_XYZ, TUPLE_XYZ]:
    grid_x = -(input.shape[0] // -block_x)
    grid_y = -(input.shape[1] // -block_x) if block_y != 1 else 1

    block = (block_x, block_y, block_z)
    grid = (grid_x, grid_y, 1)
    return block, grid


def quat_sqsum(quat: Quaternion) -> Tensor:
    quat = cast_quaternion(quat.contiguous())

    output = torch.zeros(quat.shape[0], 1,
                         dtype=quat.dtype, device=quat.device)
    block, grid = block_grid_dim(output)
    kernels.quaternion_squared_sum(output.shape[0], quat, output,
                                   block=block, grid=grid)
    return output


def quat_mag(quat: Quaternion) -> Tensor:
    quat = cast_quaternion(quat.contiguous())

    output = torch.zeros(quat.shape[0], 1,
                         dtype=quat.dtype, device=quat.device)
    block, grid = block_grid_dim(output)
    kernels.quaternion_magnitude(output.shape[0], quat, output,
                                 block=block, grid=grid)
    return output


def quat_norm(quat: Quaternion) -> Quaternion:
    quat = cast_quaternion(quat.contiguous())

    output = Quaternion(torch.zeros_like(quat))
    block, grid = block_grid_dim(output)
    kernels.quaternion_normalize(output.shape[0], quat, output,
                                 block=block, grid=grid)
    return output


def quat_conj(quat: Quaternion) -> Quaternion:
    quat = cast_quaternion(quat.contiguous())

    output = Quaternion(torch.zeros_like(quat))
    block, grid = block_grid_dim(output)
    kernels.quaternion_conjugate(output.shape[0], quat, output,
                                 block=block, grid=grid)
    return output


def quat_inv(quat: Quaternion) -> Quaternion:
    quat = cast_quaternion(quat.contiguous())

    output = Quaternion(torch.zeros_like(quat))
    block, grid = block_grid_dim(output)
    kernels.quaternion_inverse(output.shape[0], quat, output,
                               block=block, grid=grid)
    return output


def quat_dot(quat1: Quaternion, quat2: Quaternion) -> Tensor:
    quat1 = cast_quaternion(quat1.contiguous())
    quat2 = cast_quaternion(quat2.contiguous())

    output = torch.zeros(quat1.shape[0], 1,
                         dtype=quat1.dtype, device=quat1.device)
    block, grid = block_grid_dim(output)
    kernels.quaternion_dot_product(output.shape[0], quat1, quat2, output,
                                   block=block, grid=grid)
    return output


def quat_mul(left: Quaternion, right: Quaternion) -> Quaternion:
    left = cast_quaternion(left.contiguous())
    right = cast_quaternion(right.contiguous())

    output = Quaternion(torch.zeros_like(left))
    block, grid = block_grid_dim(output)
    kernels.quaternion_multiply(output.shape[0], left, right, output,
                                block=block, grid=grid)
    return output
