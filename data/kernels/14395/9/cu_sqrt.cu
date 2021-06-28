#include "includes.h"
__global__ void cu_sqrt(const float *A, float *B, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
B[tid] = sqrtf(A[tid]);
tid += stride;
}
}