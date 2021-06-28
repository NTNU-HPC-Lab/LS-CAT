#include "includes.h"
__global__ void multi_scale_kernel(const float *data_in, const float *scale, float *data_out, int width, int height) {
const int x = blockDim.x * blockIdx.x + threadIdx.x;
const int y = blockDim.y * blockIdx.y + threadIdx.y;

if (x < width && y < height) {
int index = y * width + x;
data_out[index] = data_in[index] * scale[y];
}
}