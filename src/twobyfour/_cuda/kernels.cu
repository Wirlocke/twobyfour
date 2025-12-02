#define R 0
#define I 1
#define J 2
#define K 3
#define CONJ_SIGNS {1, -1, -1, -1}
#define MUL_INDICES { \
    {R, I, J, K},     \
    {I, R, K, J},     \
    {J, K, R, I},     \
    {K, J, I, R},     \
}
#define MUL_SIGNS {  \
    {1, -1, -1, -1}, \
    {1, 1, 1, -1},   \
    {1, -1, 1, 1},   \
    {1, 1, -1, 1},   \
}

#define SQUARED_SUM ((tens[tx][R] * tens[tx][R]) + \
                     (tens[tx][I] * tens[tx][I]) + \
                     (tens[tx][J] * tens[tx][J]) + \
                     (tens[tx][K] * tens[tx][K]))

__global__ void quaternion_squared_sum(
    const size_t X_SIZE,
    Tensor<float, 2> tens,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= X_SIZE)
        return;

    output[tx][0] = SQUARED_SUM;
}

__global__ void quaternion_magnitude(
    const size_t X_SIZE,
    Tensor<float, 2> tens,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= X_SIZE)
        return;

    output[tx][0] = sqrtf(SQUARED_SUM);
}

__global__ void quaternion_normalize(
    const size_t X_SIZE,
    Tensor<float, 2> tens,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t tz = threadIdx.z;
    if (tx >= X_SIZE)
        return;

    output[tx][tz] = tens[tx][tz] / sqrtf(SQUARED_SUM);
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

    const int8_t CONJ_SI = CONJ_SIGNS;

    output[tx][tz] = tens[tx][tz] * CONJ_SI[tz];
}

__global__ void quaternion_inverse(
    const size_t X_SIZE,
    Tensor<float, 2> tens,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t tz = threadIdx.z;
    if (tx >= X_SIZE)
        return;

    const int8_t CONJ_SI = CONJ_SIGNS;

    output[tx][tz] = (tens[tx][tz] * CONJ_SI[tz]) / SQUARED_SUM;
}

__global__ void quaternion_dot_product(
    const size_t X_SIZE,
    Tensor<float, 2> tens_1,
    Tensor<float, 2> tens_2,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= X_SIZE)
        return;

    output[tx][0] =
        (tens_1[tx][R] * tens_2[tx][R]) +
        (tens_1[tx][I] * tens_2[tx][I]) +
        (tens_1[tx][J] * tens_2[tx][J]) +
        (tens_1[tx][K] * tens_2[tx][K]);
}

__global__ void quaternion_multiply(
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
