#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define VALID(tens)                        \
    TORCH_CHECK(tens.is_contiguous());     \
    TORCH_CHECK(tens.is_floating_point()); \
    TORCH_INTERNAL_ASSERT(tens.device().type() == at::DeviceType::CUDA);

#define QUAT_STRIDE 4
#define R 0
#define I 1
#define J 2
#define K 3

#define X 0
#define Y 1
#define Z 2

namespace twobyfour
{
    /*
    =============================================
    Quaternion Multiply
    =============================================
    */

    static __constant__ uint8_t INDEX[4][4] = {
        {R, I, J, K},
        {I, R, K, J},
        {J, K, R, I},
        {K, J, I, R},
    };
    static __constant__ int8_t SIGN[4][4] = {
        {1, -1, -1, -1},
        {1, 1, 1, -1},
        {1, -1, 1, 1},
        {1, 1, -1, 1},
    };

    __global__ void quaternion_multiply_kernel(
        size_t numel,
        const float *left,
        const float *right,
        float *__restrict__ result)
    {
        const size_t qx = ((blockIdx.x * blockDim.x) + threadIdx.x) * QUAT_STRIDE;
        const uint8_t idz = threadIdx.z;
        if (qx + idz >= numel)
            return;

        const float *leftid = &left[qx];
        const float *rightid = &right[qx];

        result[qx + idz] =
            (leftid[R] * rightid[INDEX[idz][R]] * SIGN[idz][R]) +
            (leftid[I] * rightid[INDEX[idz][I]] * SIGN[idz][I]) +
            (leftid[J] * rightid[INDEX[idz][J]] * SIGN[idz][J]) +
            (leftid[K] * rightid[INDEX[idz][K]] * SIGN[idz][K]);
    }

    at::Tensor quaternion_multiply_cuda(
        const at::Tensor &tens_left,
        const at::Tensor &tens_right)
    {
        TORCH_CHECK(tens_left.sizes() == tens_right.sizes())
        VALID(tens_left);
        VALID(tens_right);

        at::Tensor result = at::empty_like(tens_left);

        const float *left_ptr = tens_left.data_ptr<float>();
        const float *right_ptr = tens_right.data_ptr<float>();
        float *result_ptr = result.data_ptr<float>();

        const size_t numel = result.numel();
        TORCH_CHECK(numel % 4 == 0, "Input length must be divisible by 4.");
        if (numel == 0)
        {
            return result;
        }
        const size_t num_quats = numel / QUAT_STRIDE;
        
        const int threads = 256;
        const int threads_x = threads / QUAT_STRIDE;
        dim3 block(threads_x, 1, QUAT_STRIDE);
        dim3 grid((num_quats + threads_x - 1) / threads_x);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        quaternion_multiply_kernel<<<grid, block, 0, stream>>>(
            numel,
            left_ptr,
            right_ptr,
            result_ptr);

        return result;
    }

    /*
    =============================================
    Quaternion Apply
    =============================================
    */

    __global__ void quaternion_apply_kernel(
        size_t numel,
        const float *quat,
        const float *point,
        float *__restrict__ result)
    {
        const size_t qx = ((blockIdx.x * blockDim.x) + threadIdx.x) * QUAT_STRIDE;
        if (qx >= numel)
            return;

        const float *quat_v = &quat[qx + 1];
        const float *point_v = &point[qx + 1];
        const float cross_prod[3] = {
            (quat_v[Y] * point_v[Z]) - (quat_v[Z] * point_v[Y]),
            (quat_v[Z] * point_v[X]) - (quat_v[X] * point_v[Z]),
            (quat_v[X] * point_v[Y]) - (quat_v[Y] * point_v[X]),
        };

        result[qx + R] = point[qx + R];
        result[qx + I] = point_v[X] + (2 * quat[qx + R] * cross_prod[X]) + (2 * ((quat_v[Y] * cross_prod[Z]) - (quat_v[Z] * cross_prod[Y])));
        result[qx + J] = point_v[Y] + (2 * quat[qx + R] * cross_prod[Y]) + (2 * ((quat_v[Z] * cross_prod[X]) - (quat_v[X] * cross_prod[Z])));
        result[qx + K] = point_v[Z] + (2 * quat[qx + R] * cross_prod[Z]) + (2 * ((quat_v[X] * cross_prod[Y]) - (quat_v[Y] * cross_prod[X])));
    }

    at::Tensor quaternion_apply_cuda(
        const at::Tensor &tens_quat,
        const at::Tensor &tens_point)
    {
        TORCH_CHECK(tens_quat.sizes() == tens_point.sizes())
        VALID(tens_quat)
        VALID(tens_point)

        at::Tensor result = at::empty_like(tens_point);

        const float *quat_ptr = tens_quat.data_ptr<float>();
        const float *point_ptr = tens_point.data_ptr<float>();
        float *result_ptr = result.data_ptr<float>();

        const size_t numel = result.numel();
        TORCH_CHECK(numel % 4 == 0, "Input length must be divisible by 4.");
        if (numel == 0)
        {
            return result;
        }
        const size_t num_quats = numel / QUAT_STRIDE;

        const int threads = 256;
        const int threads_x = threads / QUAT_STRIDE;
        dim3 block(threads_x, 1, 1);
        dim3 grid((num_quats + threads_x - 1) / threads_x);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        quaternion_apply_kernel<<<grid, block, 0, stream>>>(
            numel,
            quat_ptr,
            point_ptr,
            result_ptr);
        return result;
    }

    /*
    =============================================
    Torch Library
    =============================================
    */

    TORCH_LIBRARY(twobyfour, m)
    {
        m.def("quaternion_multiply(Tensor left, Tensor right) -> Tensor");
        m.def("quaternion_apply(Tensor quat, Tensor point) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(twobyfour, CUDA, m)
    {
        m.impl("quaternion_multiply", &quaternion_multiply_cuda);
        m.impl("quaternion_apply", &quaternion_apply_cuda);
    }
}