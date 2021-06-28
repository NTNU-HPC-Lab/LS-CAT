#include "includes.h"
__global__ void cu_addition(const double *A, const double *B, double *C, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
C[tid] = __fadd_rd(A[tid], B[tid]);
tid += stride;
}
}