#include "includes.h"
__global__ void TemporalConvolutionTBC_fp_bias( float* output_features, float* bias, int output_stride, int rows) {
int x = blockIdx.x * 32 + threadIdx.x;
float b = bias[x];
for (int row = blockIdx.y; row < rows; row += gridDim.y) {
output_features[row * output_stride + x] = b;
}
}