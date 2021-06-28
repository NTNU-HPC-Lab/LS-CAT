#include "includes.h"
__global__ void cu_dleaky_relu(const float* src, float* dst, int n){
const float leaky_relu_alpha = 100.0;
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
float p = 0.0;
float n = 0.0;
if(src[tid] > 0.0) p = 1;
if(src[tid] < 0.0) n = 1;
n = fdividef(n, leaky_relu_alpha);
dst[tid] = __fadd_rd(p, n);
tid += stride;
}
}