# Copyright (c) 2025 Marisha Norcross
# Copyright (c) 2023 Chaoyang Wang
#
# This source code contains modifications of work covered by MIT license.
# See LICENSE and LICENSE-dqtorch for the full license text.

import cutex

KERNELS = """
//cuda
__global__ void kernel_quaternion_mul(
    Tensor<float, 2> inputs_1,
    Tensor<float, 2> inputs_2,
    Tensor<float, 2> outputs,
    uint32_t B,
    uint8_t D1,
    uint8_t D2)
{
    const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= B)
		return;
    
    float aw = 0, ax = 0, ay = 0, az = 0;
    float bw = 0, bx = 0, by = 0, bz = 0;
    if (D1 == 3)
	{
		aw = 0, ax = inputs_1[b][0], ay = inputs_1[b][1], az = inputs_1[b][2];
	}
	else
	{
		aw = inputs_1[b][0], ax = inputs_1[b][1], ay = inputs_1[b][2], az = inputs_1[b][3];
	}

	if (D2 == 3)
	{
		bw = 0, bx = inputs_2[b][0], by = inputs_2[b][1], bz = inputs_2[b][2];
	}
	else
	{
		bw = inputs_2[b][0], bx = inputs_2[b][1], by = inputs_2[b][2], bz = inputs_2[b][3];
	}

	outputs[b][0] = aw * bw - ax * bx - ay * by - az * bz;
	outputs[b][1] = aw * bx + ax * bw + ay * bz - az * by;
	outputs[b][2] = aw * by - ax * bz + ay * bw + az * bx;
	outputs[b][3] = aw * bz + ax * by - ay * bx + az * bw;
}


__global__ void kernel_quaternion_mul_backward(
    Tensor<float, 2> grad,
    uint32_t B,
    uint8_t D1,
    uint8_t D2,
    Tensor<float, 2> inputs_1,
    Tensor<float, 2> inputs_2,
    Tensor<float, 2> grad_inputs_1,
    Tensor<float, 2> grad_inputs_2)
{
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t b = t / 4;
    if (b >= B)
        return;
    const uint8_t d = t - b * 4;
    
    float aw = 0, ax = 0, ay = 0, az = 0;
    float bw = 0, bx = 0, by = 0, bz = 0;
    if (D1 == 3)
	{
		aw = 0, ax = inputs_1[b][0], ay = inputs_1[b][1], az = inputs_1[b][2];
	}
	else
	{
		aw = inputs_1[b][0], ax = inputs_1[b][1], ay = inputs_1[b][2], az = inputs_1[b][3];
	}

	if (D2 == 3)
	{
		bw = 0, bx = inputs_2[b][0], by = inputs_2[b][1], bz = inputs_2[b][2];
	}
	else
	{
		bw = inputs_2[b][0], bx = inputs_2[b][1], by = inputs_2[b][2], bz = inputs_2[b][3];
	}
    
    if (d == 0)
	{
		if (D1 > 3)
		{
			grad_inputs_1[b][d] = grad[b][0] * bw + grad[b][1] * bx + grad[b][2] * by + grad[b][3] * bz;
		}
		if (D2 > 3)
		{
			grad_inputs_2[b][d] = grad[b][0] * aw + grad[b][1] * ax + grad[b][2] * ay + grad[b][3] * az;
		}
	}
	else if (d == 1)
	{
		grad_inputs_1[b][d] = grad[b][0] * (-bx) + grad[b][1] * bw + grad[b][2] * (-bz) + grad[b][3] * by;
		grad_inputs_2[b][d] = grad[b][0] * (-ax) + grad[b][1] * aw + grad[b][2] * az + grad[b][3] * (-ay);
	}
	else if (d == 2)
	{
		grad_inputs_1[b][d] = grad[b][0] * (-by) + grad[b][1] * bz + grad[b][2] * bw + grad[b][3] * (-bx);
		grad_inputs_2[b][d] = grad[b][0] * (-ay) + grad[b][1] * (-az) + grad[b][2] * aw + grad[b][3] * ax;
	}
	else
	{
		grad_inputs_1[b][d] = grad[b][0] * (-bz) + grad[b][1] * (-by) + grad[b][2] * bx + grad[b][3] * bw;
		grad_inputs_2[b][d] = grad[b][0] * (-az) + grad[b][1] * ay + grad[b][2] * (-ax) + grad[b][3] * aw;
	}
}


__global__ void kernel_quaternion_mul_backward_backward(
    Tensor<float, 2> grad_out_1,
    Tensor<float, 2> grad_out_2,
    uint32_t B,
    uint8_t D1,
    uint8_t D2,
    Tensor<float, 2> grad,
    Tensor<float, 2> inputs_1,
    Tensor<float, 2> inputs_2,
    Tensor<float, 2> grad_grad,
    Tensor<float, 2> grad_grad_inputs_1,
    Tensor<float, 2> grad_grad_inputs_2)
{
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t b = t / 4;
    if (b >= B)
		return;
    const uint8_t d = t - b * 4;
    
    float aw = 0, ax = 0, ay = 0, az = 0;
    float bw = 0, bx = 0, by = 0, bz = 0;
    
    float d_aw = 0, d_ax = 0, d_ay = 0, d_az = 0;
    float d_bw = 0, d_bx = 0, d_by = 0, d_bz = 0;
    
    if (D1 == 3)
	{
		aw = 0, ax = inputs_1[b][0], ay = inputs_1[b][1], az = inputs_1[b][2];
		d_aw = 0, d_ax = grad_out_1[b][0], d_ay = grad_out_1[b][1], d_az = grad_out_1[b][2];
	}
	else
	{
		aw = inputs_1[b][0], ax = inputs_1[b][1], ay = inputs_1[b][2], az = inputs_1[b][3];
		d_aw = grad_out_1[b][0], d_ax = grad_out_1[b][1], d_ay = grad_out_1[b][2], d_az = grad_out_1[b][3];
	}

	if (D2 == 3)
	{
		bw = 0, bx = inputs_2[b][0], by = inputs_2[b][1], bz = inputs_2[b][2];
		d_bw = 0, d_bx = grad_out_2[b][0], d_by = grad_out_2[b][1], d_bz = grad_out_2[b][2];
	}
	else
	{
		bw = inputs_2[b][0], bx = inputs_2[b][1], by = inputs_2[b][2], bz = inputs_2[b][3];
		d_bw = grad_out_2[b][0], d_bx = grad_out_2[b][1], d_by = grad_out_2[b][2], d_bz = grad_out_2[b][3];
	}
    
    if (d == 0)
	{
		if (D1 > 3)
		{
			grad_grad_inputs_1[b][d] = d_bw * grad[b][0] + d_bx * grad[b][1] + d_by * grad[b][2] + d_bz * grad[b][3];
		}
		if (D2 > 3)
		{
			grad_grad_inputs_2[b][d] = d_aw * grad[b][0] + d_ax * grad[b][1] + d_ay * grad[b][2] + d_az * grad[b][3];
		}
		grad_grad[b][d] = d_aw * bw + d_bw * aw - d_ax * bx - d_bx * ax - d_ay * by - d_by * ay - d_az * bz - d_bz * az;
	}
	else if (d == 1)
	{
		grad_grad_inputs_1[b][d] = d_bw * grad[b][1] - d_bx * grad[b][0] + d_by * grad[b][3] - d_bz * grad[b][2];
		grad_grad_inputs_2[b][d] = d_aw * grad[b][1] - d_ax * grad[b][0] - d_ay * grad[b][3] + d_az * grad[b][2];

		grad_grad[b][d] = d_aw * bx + d_bw * ax + d_ax * bw + d_bx * aw + d_ay * bz - d_by * az - d_az * by + d_bz * ay;
	}
	else if (d == 2)
	{
		grad_grad_inputs_1[b][d] = d_bw * grad[b][2] - d_bx * grad[b][3] - d_by * grad[b][0] + d_bz * grad[b][1];
		grad_grad_inputs_2[b][d] = d_aw * grad[b][2] + d_ax * grad[b][3] - d_ay * grad[b][0] - d_az * grad[b][1];

		grad_grad[b][d] = d_aw * by + d_bw * ay - d_ax * bz + d_bx * az + d_ay * bw + d_by * aw + d_az * bx - d_bz * ax;
	}
	else
	{
		grad_grad_inputs_1[b][d] = d_bw * grad[b][3] + d_bx * grad[b][2] - d_by * grad[b][1] - d_bz * grad[b][0];
		grad_grad_inputs_2[b][d] = d_aw * grad[b][3] - d_ax * grad[b][2] + d_ay * grad[b][1] - d_az * grad[b][0];

		grad_grad[b][d] = d_aw * bz + d_bw * az + d_ax * by - d_bx * ay - d_ay * bx + d_by * ax + d_az * bw + d_bz * aw;
	}
}

__global__ void kernel_quaternion_magnitude(
	Tensor<float, 2> inputs,
    uint32_t B,
    uint8_t D,
    Tensor<float, 2> outputs)
{
    const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
    
	if (b >= B)
		return;
    printf("ThreadID: %d\\nBlockID: %d\\nBlockDim: %d\\n", threadIdx.x, blockIdx.x, blockDim.x);
        
    float w = 0, x = 0, y = 0, z = 0;
    
    if (D == 3)
    {
		w = 0, x = inputs[b][0], y = inputs[b][1], z = inputs[b][2];
	}
    else
    {
		w = inputs[b][0], x = inputs[b][1], y = inputs[b][2], z = inputs[b][3];
	}
    
    outputs[b][0] = sqrtf((w*w) + (x*x) + (y*y) + (z*z));
}

__global__ void kernel_quaternion_conjugate(
    Tensor<float, 2> inputs,
    uint32_t B,
    Tensor<float, 2> outputs)
{
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t b = t / 4;
    if (b >= B)
		return;
    const uint8_t d = t - b * 4;
    
    if (d == 0)
    {
        outputs[b][d] = inputs[b][d];
    }
    else
    {
        outputs[b][d] = -inputs[b][d];
    }
}
//!cuda
"""

kernels = cutex.SourceModule(KERNELS, float_bits=32, boundscheck=False)
