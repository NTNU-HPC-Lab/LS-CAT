#include "includes.h"
__global__ void mul(int * A, int * B, int * C){
int i = blockIdx.x;
int j = threadIdx.x;
C[i * N + j] = 0;
for (int k = 0; k < N; k++){
C[i * N + j] += A[i * N + k] * B[k * N + j];
}
}