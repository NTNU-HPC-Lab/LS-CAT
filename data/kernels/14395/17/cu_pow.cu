#include "includes.h"
__global__ void cu_pow(const float* src, float* dst, const float power, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
dst[tid] = powf(src[tid], power);
tid += stride;
}
}