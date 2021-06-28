#include "includes.h"
__global__ void cu_sigmoid(double* src, double* dst, int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
double tmp = __fmul_rd(src[tid], -1.0);
tmp = __expf(tmp);
tmp = __fadd_rd(tmp, 1.0);
dst[tid] = __fdividef(1.0, tmp);
tid += stride;
}
}