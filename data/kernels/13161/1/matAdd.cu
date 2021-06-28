#include "includes.h"
__global__ void matAdd(float *A, float *B, float *C, int N){
// Las matrices se recorren con la ordenacion de Fortran
int j = blockIdx.x * blockDim.x + threadIdx.x; // indice filas
int i = blockIdx.y * blockDim.y + threadIdx.y; // indice columnas
int tid = (i * N) + j;

if(i < N && j < N)
C[tid] = A[tid] + B[tid];
}