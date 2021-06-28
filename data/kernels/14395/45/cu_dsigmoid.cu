#include "includes.h"
__global__ void cu_dsigmoid(const float* src, float* dst, int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
float tmp = __expf(src[tid]);
float tmp2 = __fadd_rd(tmp, 1.0);
tmp2 = __fmul_rd(tmp2, tmp2);
dst[tid] = fdividef(tmp, tmp2);
tid += stride;
}
}