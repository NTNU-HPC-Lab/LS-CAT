#include "includes.h"
__global__ void cu_dtanh(const float* src, float* dst, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
float tmp = __fmul_rd(src[tid], src[tid]);
dst[tid] = __fsub_rd(1.0, tmp);
tid += stride;
}
}