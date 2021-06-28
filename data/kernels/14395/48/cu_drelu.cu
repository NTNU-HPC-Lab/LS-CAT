#include "includes.h"
__global__ void cu_drelu(const float* src, float* dst, int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
if(src[tid] > 0.0) dst[tid] = 1.0;
else dst[tid] = 0.0;
tid += stride;
}
}