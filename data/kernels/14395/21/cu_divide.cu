#include "includes.h"
__global__ void cu_divide(const float* numerator, const float* denominator, float* dst, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
if(0 == denominator[tid]) dst[tid] = 0.0;
else dst[tid] = __fdividef(numerator[tid], denominator[tid]);
tid += stride;
}
}