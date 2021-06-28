#include "includes.h"
__global__ void relu_ker(float* src, float* dst, int N){
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i >= N){
return;
}
dst[i] = fmaxf(0.0, src[i]);
}