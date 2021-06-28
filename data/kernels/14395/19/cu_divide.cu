#include "includes.h"
__global__ void cu_divide(const float* numerator, float* dst, const float denominator, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
if(0 == denominator) dst[tid] = 0.0;
else dst[tid] = __fdividef(numerator[tid], denominator);
tid += stride;
}
}