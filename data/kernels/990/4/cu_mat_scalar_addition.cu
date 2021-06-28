#include "includes.h"
__global__ void cu_mat_scalar_addition(double *A, const double b, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
A[tid] = __fadd_rd(A[tid], b);
tid += stride;
}
}