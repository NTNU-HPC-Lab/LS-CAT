#include "includes.h"
__global__ void Kernel11(int N, int M, int P, float *A, float *B, float *C) {

__shared__ float sA[SIZE][SIZE];
__shared__ float sB[SIZE][SIZE];

int bx = blockIdx.x;  int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int row = by * SIZE + ty;
int col = bx * SIZE + tx;
int m, k, iter;



float tmp = 0.0;
iter = P%SIZE;
if (iter == 0) {
for (m=0; m < P; m=m+SIZE) {
sA[ty][tx] = A[row*P + m + tx];
sB[ty][tx] = B[col + (m + ty)*M];
__syncthreads();
for (k=0; k<SIZE; k++)
tmp += sA[ty][k] * sB[k][tx];
__syncthreads();
}
}
else {
for (m=0; m < P-iter; m=m+SIZE) {
sA[ty][tx] = A[row*P + m + tx];
sB[ty][tx] = B[col + (m + ty)*M];
__syncthreads();
for (k=0; k<SIZE; k++)
tmp += sA[ty][k] * sB[k][tx];
__syncthreads();
}

if (col < P && row < N) sA[ty][tx] = A[row*P + m + tx];   else sA[ty][tx] = 0.0;
if (row < P && col < M) sB[ty][tx] = B[col + (m + ty)*M]; else sB[ty][tx] = 0.0;
__syncthreads();
for (k=0; k<iter; k++)
tmp += sA[ty][k] * sB[k][tx];
}
if ((row < N) && (col < M)) C[row*M+col] = tmp;

}