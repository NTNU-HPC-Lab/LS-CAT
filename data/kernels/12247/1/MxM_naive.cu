#include "includes.h"
__global__ void MxM_naive(double* A, double* B, double* C, const int N) {

int i = blockIdx.y * blockDim.y + threadIdx.y;    // Row i of matrix C
int j = blockIdx.x * blockDim.x + threadIdx.x;    // Column j of matrix C

double C_temp = 0;
for (int k=0; k<N; k++) {
// use 1D indexing
C_temp += A[i*N + k] * B[k*N + j];
}

// write back to global memory
// no synchronization needed here because one thread handles one element
C[i*N + j] = C_temp;

}