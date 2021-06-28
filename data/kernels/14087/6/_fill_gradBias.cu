#include "includes.h"
__global__ void _fill_gradBias(float *gradBias, const float *gradOutput, float scale, int batch_n, int output_n, int output_h, int output_w) {
gradOutput += blockIdx.x*output_h*output_w;
__shared__ float shGrad[128]; // 32*4
float g = .0f;
int oz,oxy;
for (oz = threadIdx.y; oz < batch_n; oz += 4) {
const float *out = gradOutput + oz*output_n*output_h*output_w;
for (oxy = threadIdx.x; oxy < output_h*output_w; oxy += 32) {
g += out[oxy];
}
}
shGrad[threadIdx.y*blockDim.x+threadIdx.x] = g;
__syncthreads();

// reduce
if (threadIdx.x == 0) {
g = .0f;
for (oxy = 0; oxy < 128; ++oxy)
g += shGrad[oxy];
gradBias[blockIdx.x] = scale*g;
}
}