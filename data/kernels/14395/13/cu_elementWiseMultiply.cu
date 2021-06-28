#include "includes.h"
__global__ void cu_elementWiseMultiply(const float *A, const float B, float *C, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
C[tid] = __fmul_rd(A[tid], B);
tid += stride;
}
}