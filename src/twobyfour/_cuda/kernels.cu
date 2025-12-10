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

    const int8_t CONJ_SI[4] = CONJ_SIGNS;

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

    const int8_t CONJ_SI[4] = CONJ_SIGNS;

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

    const size_t tx1 = min(tx, tens_1.size(0) - 1);
    const size_t tx2 = min(tx, tens_2.size(0) - 1);

    output[tx][0] =
        (tens_1[tx1][R] * tens_2[tx2][R]) +
        (tens_1[tx1][I] * tens_2[tx2][I]) +
        (tens_1[tx1][J] * tens_2[tx2][J]) +
        (tens_1[tx1][K] * tens_2[tx2][K]);
}

__global__ void quaternion_multiply(
    const size_t X_SIZE,
    Tensor<float, 2> tens_left,
    Tensor<float, 2> tens_right,
    Tensor<float, 2> output)
{
    const size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint8_t tz = threadIdx.z;
    if (tx >= X_SIZE)
        return;

    const size_t txl = min(tx, tens_left.size(0) - 1);
    const size_t txr = min(tx, tens_right.size(0) - 1);

    const uint8_t IN[4][4] = MUL_INDICES;
    const int8_t SI[4][4] = MUL_SIGNS;

    output[tx][tz] =
        (tens_left[txl][R] * tens_right[txr][IN[tz][R]] * SI[tz][R]) +
        (tens_left[txl][I] * tens_right[txr][IN[tz][I]] * SI[tz][I]) +
        (tens_left[txl][J] * tens_right[txr][IN[tz][J]] * SI[tz][J]) +
        (tens_left[txl][K] * tens_right[txr][IN[tz][K]] * SI[tz][K]);
}
