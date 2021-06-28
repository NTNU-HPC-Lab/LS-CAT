#include "includes.h"
__global__ void mul(int* A, int* B, int* C){
int col = blockIdx.x * blockDim.x + threadIdx.x;
int lig = blockIdx.y * blockDim.y + threadIdx.y;

int index = lig * N + col;

if (col < N && lig < N){
int inter = 0;
for (int i = 0; i<N; ++i){
inter += A[lig*N + i] * B[i*N + col];
}
C[index] = inter;
}
}