#include "includes.h"
__device__ float logit1(const float x) {
return expf(x) / (1. + expf(x));
}
__global__ void logit(float* y, const float* x, int leng) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < leng) {
y[ i ] = logit1(x[ i ]);
}
}