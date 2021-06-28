#include "includes.h"
__global__ void cu_mat_scalar_divide(double *A, double B, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
A[tid] = __fdiv_rd(A[tid], B);
tid += stride;
}
}