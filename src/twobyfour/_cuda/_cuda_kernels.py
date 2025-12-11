# Copyright (c) 2025 Marisha Norcross
#
# This source code is covered by MIT license
# See LICENSE for the full license text.


from typing import cast
from pathlib import Path

import cutex
import torch
from torch import Tensor, Size

from ..typing import Quaternion


TUPLE_XYZ = tuple[int, int, int]


DIR = Path(__file__).parent / "kernels.cu"
with open(DIR, 'r') as file:
    KERNELS = file.read()
kernels = cutex.SourceModule(KERNELS, float_bits=32, boundscheck=False)


def flatten(input: Quaternion) -> tuple[Quaternion, Size]:
    out_shape = input.shape
    return cast(Quaternion, input.contiguous().view(-1, 4)), out_shape


def block_grid_dim(input: Tensor, block_x=256, block_y=1, block_z=4) -> tuple[TUPLE_XYZ, TUPLE_XYZ]:
    grid_x = -(input.shape[0] // -block_x)
    grid_y = -(input.shape[1] // -block_x) if block_y != 1 else 1

    block = (block_x, block_y, block_z)
    grid = (grid_x, grid_y, 1)
    return block, grid


def quat_mul(left: Quaternion, right: Quaternion) -> Quaternion:
    left, shape_left = flatten(left)
    right, shape_right = flatten(right)

    out_shape = torch.broadcast_shapes(shape_left, shape_right)
    output = torch.empty(out_shape,
                         dtype=left.dtype, device=left.device).view(-1, 4)

    block, grid = block_grid_dim(output)
    kernels.quaternion_multiply(out_shape[0], left, right, output,
                                block=block, grid=grid)

    return Quaternion(output.view(out_shape))
