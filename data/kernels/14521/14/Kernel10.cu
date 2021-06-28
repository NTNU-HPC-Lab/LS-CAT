#include "includes.h"
__global__ void Kernel10(int N, int M, int P, float *A, float *B, float *C) {

__shared__ float sA[SIZE][SIZE];
__shared__ float sB[SIZE][SIZE];

int bx = blockIdx.x;  int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int row = by * SIZE + ty;
int col = bx * SIZE + tx;

float tmp = 0.0;
for (int m=0; m < P; m=m+SIZE) {
sA[ty][tx] = A[row*P + m + tx];
sB[ty][tx] = B[col + (m + ty)*M];
__syncthreads();
for (int k=0; k<SIZE; k++)
tmp += sA[ty][k] * sB[k][tx];
__syncthreads();
}
C[row*M+col] = tmp;
}