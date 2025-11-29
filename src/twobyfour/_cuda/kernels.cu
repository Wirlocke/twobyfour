#define R 0
#define I 1
#define J 2
#define K 3
#define MUL_INDICES { \
    {0, 1, 2, 3},     \
    {1, 0, 3, 2},     \
    {2, 3, 0, 1},     \
    {3, 2, 1, 0}}
#define MUL_SIGNS {  \
    {1, -1, -1, -1}, \
    {1, 1, 1, -1},   \
    {1, -1, 1, 1},   \
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

__global__ void quaternion_inverse(
    const size_t X_SIZE,
    Tensor<float, 2> tens,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t tz = threadIdx.z;
    if (tx >= X_SIZE)
        return;

    const uint8_t VEC_SIGN = (1 - (2 * (tz > 0)));
    const float DIVISOR = ((tens[tx][R] * tens[tx][R]) +
                           (tens[tx][I] * tens[tx][I]) +
                           (tens[tx][J] * tens[tx][J]) +
                           (tens[tx][K] * tens[tx][K]));

    output[tx][tz] = (tens[tx][tz] * VEC_SIGN) / DIVISOR;
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