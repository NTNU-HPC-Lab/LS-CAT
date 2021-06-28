#include "includes.h"
__global__ void cu_plus(const float *A, const float *B, float *C, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
C[tid] = __fadd_rd(A[tid], B[tid]);
tid += stride;
}
}