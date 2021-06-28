#include "includes.h"
__global__ void cu_minus(const float *A, float *B, const float c, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
B[tid] = __fsub_rd(A[tid], c);
tid += stride;
}
}