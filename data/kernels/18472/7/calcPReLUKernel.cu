#include "includes.h"
__global__ void calcPReLUKernel(const float *input, float *output, const float *weights, int width, int height, int channels)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
if (x >= width || y >= height) {
return;
}

output[y * width + x] = input[y * width + x] > 0 ? input[y * width + x] : input[y * width + x] * weights[y % channels];

}