#include "includes.h"
__global__ void matSum(float* S, float* A, float* B, int N) {
int j = blockIdx.y*blockDim.y + threadIdx.y;
int i = blockIdx.x*blockDim.x + threadIdx.x;
int tid = i*N + j;
if (tid < N*N) {
S[tid] = A[tid] + B[tid];
}
}