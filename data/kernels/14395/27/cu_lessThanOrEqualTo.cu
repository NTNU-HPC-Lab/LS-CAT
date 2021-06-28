#include "includes.h"
__global__ void cu_lessThanOrEqualTo(const float* src, float* dst, const float val, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
if(src[tid] <= val) dst[tid] = 1.0;
else dst[tid] = 0.0;
tid += stride;
}
}