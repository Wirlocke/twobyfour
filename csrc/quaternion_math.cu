#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define VALID(TENSOR)                                               \
    TORCH_INTERNAL_ASSERT(TENSOR.is_cuda());                        \
    TORCH_CHECK(TENSOR.is_contiguous());                            \
    TORCH_CHECK(TENSOR.is_floating_point() || TENSOR.is_complex()); \
    TORCH_CHECK(TENSOR.numel() != 0)                                \
    TORCH_CHECK(TENSOR.numel() % 4 == 0, "Input length must be divisible by 4.");

#define DISPATCH_DTYPE(TENSOR, KERNEL, ...)      \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2( \
        at::ScalarType::Half,                    \
        at::ScalarType::BFloat16,                \
        TENSOR.scalar_type(),                    \
        #KERNEL,                                 \
        [&]() { KERNEL<scalar_t> __VA_ARGS__; })

#define IMPLEMENT(M, FUNCTION) M.impl(#FUNCTION, &FUNCTION)

#define THREADS 256
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
    cuda::std::pair<dim3, dim3> quat_threads(
        const size_t numel,
        const int threads_z = QUAT_STRIDE)
    {
        const size_t num_quats = numel / QUAT_STRIDE;
        const int threads_x = THREADS / QUAT_STRIDE;
        dim3 grid((num_quats + threads_x - 1) / threads_x);
        dim3 block(threads_x, 1, threads_z);
        return cuda::std::make_pair(grid, block);
    }

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
    static __constant__ int SIGN[4][4] = {
        {1, -1, -1, -1},
        {1, 1, 1, -1},
        {1, -1, 1, 1},
        {1, 1, -1, 1},
    };

    template <typename scalar_t>
    __global__ void quaternion_multiply_kernel(
        const size_t numel,
        const scalar_t *left,
        const scalar_t *right,
        scalar_t *__restrict__ result)
    {
        const size_t qx = ((blockIdx.x * blockDim.x) + threadIdx.x) * QUAT_STRIDE;
        const uint8_t idz = threadIdx.z;
        if (qx + idz >= numel)
            return;

        const scalar_t *leftid = &left[qx];
        const scalar_t *rightid = &right[qx];

        result[qx + idz] =
            (leftid[R] * rightid[INDEX[idz][R]] * SIGN[idz][R]) +
            (leftid[I] * rightid[INDEX[idz][I]] * SIGN[idz][I]) +
            (leftid[J] * rightid[INDEX[idz][J]] * SIGN[idz][J]) +
            (leftid[K] * rightid[INDEX[idz][K]] * SIGN[idz][K]);
    }

    at::Tensor quaternion_multiply(
        const at::Tensor &tens_left,
        const at::Tensor &tens_right)
    {
        TORCH_CHECK(tens_left.sizes() == tens_right.sizes())
        VALID(tens_left);
        VALID(tens_right);

        at::Tensor result = at::empty_like(tens_left);
        const size_t numel = result.numel();
        auto [grid, block] = quat_threads(numel);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        DISPATCH_DTYPE(
            result,
            quaternion_multiply_kernel,
            <<<grid, block, 0, stream>>>(
                numel,
                tens_left.data_ptr<scalar_t>(),
                tens_right.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>()));

        return result;
    }

    /*
    =============================================
    Quaternion Apply
    =============================================
    */

    template <typename scalar_t>
    __global__ void quaternion_unit_apply_kernel(
        const size_t numel,
        const scalar_t *quat,
        const scalar_t *point,
        scalar_t *__restrict__ result)
    {
        const size_t qx = ((blockIdx.x * blockDim.x) + threadIdx.x) * QUAT_STRIDE;
        if (qx >= numel)
            return;

        const scalar_t *quat_v = &quat[qx + 1];
        const scalar_t *point_v = &point[qx + 1];
        const scalar_t cross_prod[3] = {
            (quat_v[Y] * point_v[Z]) - (quat_v[Z] * point_v[Y]),
            (quat_v[Z] * point_v[X]) - (quat_v[X] * point_v[Z]),
            (quat_v[X] * point_v[Y]) - (quat_v[Y] * point_v[X]),
        };

        result[qx + R] = point[qx + R];
        result[qx + I] = point_v[X] + (2 * quat[qx + R] * cross_prod[X]) + (2 * ((quat_v[Y] * cross_prod[Z]) - (quat_v[Z] * cross_prod[Y])));
        result[qx + J] = point_v[Y] + (2 * quat[qx + R] * cross_prod[Y]) + (2 * ((quat_v[Z] * cross_prod[X]) - (quat_v[X] * cross_prod[Z])));
        result[qx + K] = point_v[Z] + (2 * quat[qx + R] * cross_prod[Z]) + (2 * ((quat_v[X] * cross_prod[Y]) - (quat_v[Y] * cross_prod[X])));
    }

    at::Tensor quaternion_unit_apply(
        const at::Tensor &tens_quat,
        const at::Tensor &tens_point)
    {
        TORCH_CHECK(tens_quat.sizes() == tens_point.sizes())
        VALID(tens_quat)
        VALID(tens_point)

        at::Tensor result = at::empty_like(tens_point);
        const size_t numel = result.numel();
        auto [grid, block] = quat_threads(numel, 1);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        DISPATCH_DTYPE(
            result,
            quaternion_unit_apply_kernel,
            <<<grid, block, 0, stream>>>(
                numel,
                tens_quat.data_ptr<scalar_t>(),
                tens_point.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>()));

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
        m.def("quaternion_unit_apply(Tensor quat, Tensor point) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(twobyfour, CUDA, m)
    {
        IMPLEMENT(m, quaternion_multiply);
        IMPLEMENT(m, quaternion_unit_apply);
    }
}