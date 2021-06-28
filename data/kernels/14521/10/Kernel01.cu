#include "includes.h"
__global__ void Kernel01 (int N, int M, int P, float *A, float *B, float *C) {

int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < N && col < M) {
float tmp = 0.0;
for (int k=0; k<P; k++)
tmp += A[row*P+k] * B[k*N+col];
C[row*N+col] = tmp;
}
}