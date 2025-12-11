#define R 0
#define I 1
#define J 2
#define K 3
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

__constant__ int INDEX[4][4] = MUL_INDICES;
__constant__ int SIGN[4][4] = MUL_SIGNS;

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

    output[tx][tz] =
        (tens_left[txl][R] * tens_right[txr][INDEX[tz][R]] * SIGN[tz][R]) +
        (tens_left[txl][I] * tens_right[txr][INDEX[tz][I]] * SIGN[tz][I]) +
        (tens_left[txl][J] * tens_right[txr][INDEX[tz][J]] * SIGN[tz][J]) +
        (tens_left[txl][K] * tens_right[txr][INDEX[tz][K]] * SIGN[tz][K]);
}
