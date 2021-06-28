#include "includes.h"

using namespace std;



__global__ void matrixMultiplicationKernel(long* A, long* B, long* C, long N) {

long ROW = (blockIdx.y*blockDim.y) + threadIdx.y;
long COL = (blockIdx.x*blockDim.x) + threadIdx.x;

long tmpSum = 0;

if (ROW < N && COL < N) {
// each thread computes one element of the block sub-matrix
for (long i = 0; i < N; i++) {
tmpSum += A[ROW * N + i] * B[i * N + COL];
}

C[ROW * N + COL] = tmpSum;
}
}