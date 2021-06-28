#include "includes.h"
__global__ void cu_tanh(const float* src, float* dst, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
dst[tid] = tanhf(src[tid]);
tid += stride;
}
}