#include "includes.h"
__global__ void cu_log(const float* src, float* dst, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
dst[tid] = __logf(src[tid]);
tid += stride;
}
}