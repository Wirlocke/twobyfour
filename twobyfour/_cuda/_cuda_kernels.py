# Copyright (c) 2025 Marisha Norcross
#
# This source code is covered by MIT license
# See LICENSE for the full license text.

import cutex
import torch
from torch import Tensor

KERNELS = """
//cuda
#define R 0
#define I 1
#define J 2
#define K 3
#define MUL_INDICES { \\
    {0, 1, 2, 3},     \\
    {1, 0, 3, 2},     \\
    {2, 3, 0, 1},     \\
    {3, 2, 1, 0}}
#define MUL_SIGNS {  \\
    {1, -1, -1, -1}, \\
    {1, 1, 1, -1},   \\
    {1, -1, 1, 1},   \\
    {1, 1, -1, 1}}

__global__ void quaternion_mul(
    const size_t X_SIZE,
    Tensor<float, 2> tens_1,
    Tensor<float, 2> tens_2,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t tz = threadIdx.z;
    if (tx >= X_SIZE)
        return;

    const uint8_t IN[4][4] = MUL_INDICES;
    const int8_t SI[4][4] = MUL_SIGNS;

    output[tx][tz] =
        (tens_1[tx][R] * tens_2[tx][IN[tz][R]] * SI[tz][R]) +
        (tens_1[tx][I] * tens_2[tx][IN[tz][I]] * SI[tz][I]) +
        (tens_1[tx][J] * tens_2[tx][IN[tz][J]] * SI[tz][J]) +
        (tens_1[tx][K] * tens_2[tx][IN[tz][K]] * SI[tz][K]);
}

__global__ void quaternion_conjugate(
    const size_t X_SIZE,
    Tensor<float, 2> tens,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t tz = threadIdx.z;
    if (tx >= X_SIZE)
        return;

    output[tx][tz] = tens[tx][tz] * (1 - (2 * (tz > 0)));
}

__global__ void quaternion_magnitude(
    const size_t X_SIZE,
    Tensor<float, 2> tens,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= X_SIZE)
        return;

    output[tx][0] = sqrtf(
        (tens[tx][R] * tens[tx][R]) +
        (tens[tx][I] * tens[tx][I]) +
        (tens[tx][J] * tens[tx][J]) +
        (tens[tx][K] * tens[tx][K]));
}
//!cuda
"""

kernels = cutex.SourceModule(KERNELS, float_bits=32, boundscheck=False)

TUPLE_XYZ = tuple[int, int, int]


def block_grid_dim(input: Tensor, block_x=256, block_y=1, block_z=4) -> tuple[TUPLE_XYZ, TUPLE_XYZ]:
    grid_x = -(input.shape[0] // -block_x)
    grid_y = -(input.shape[1] // -block_x) if block_y != 1 else 1

    block = (block_x, block_y, block_z)
    grid = (grid_x, grid_y, 1)
    return block, grid


def quat_mul(left: Tensor, right: Tensor) -> Tensor:
    output = torch.zeros_like(left)
    block, grid = block_grid_dim(output)
    kernels.quaternion_mul(output.shape[0], left, right, output,
                           block=block, grid=grid)
    return output


def quat_conj(quat: Tensor) -> Tensor:
    output = torch.zeros_like(quat)
    block, grid = block_grid_dim(output)
    kernels.quaternion_conjugate(output.shape[0], quat, output,
                                 block=block, grid=grid)
    return output


def quat_mul_grad(grad: Tensor, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
    left_grad = quat_mul(grad, quat_conj(right))
    right_grad = quat_mul(quat_conj(left), grad)
    return left_grad, right_grad


def quat_mul_second_grad(grad: Tensor, left: Tensor, right: Tensor, left_grad: Tensor, right_grad: Tensor) -> tuple[Tensor, Tensor]:
    left_second_grad = quat_mul(left_grad, quat_conj(right)) + \
        quat_mul(grad, quat_conj(right_grad))

    right_second_grad = quat_mul(quat_conj(left), right_grad) + \
        quat_mul(quat_conj(grad), right)

    return left_second_grad, right_second_grad


def quat_mag(quat: Tensor) -> Tensor:
    output = torch.zeros_like(quat).reshape(quat.shape[0], 1)
    block, grid = block_grid_dim(output)
    kernels.quaternion_magnitude(output.shape[0], quat, output,
                                 block=block, grid=grid)
    return output
