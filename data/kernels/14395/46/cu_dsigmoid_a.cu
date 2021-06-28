#include "includes.h"
__global__ void cu_dsigmoid_a(const float* src, float* dst, int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
float tmp = __fsub_rd(1.0, src[tid]);
dst[tid] = __fmul_rd(tmp, src[tid]);
tid += stride;
}
}