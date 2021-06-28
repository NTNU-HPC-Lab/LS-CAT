#include "includes.h"
__global__ void TemporalConvolutionTBC_bp_bias( float* matrix, float* target, int rows, int stride, float scale) {
int i = blockIdx.x * 32 + threadIdx.x;
float t = 0;
for (int j = blockIdx.y; j < rows; j += gridDim.y)
t += matrix[j * stride + i];
atomicAdd(&target[i], t * scale);
}