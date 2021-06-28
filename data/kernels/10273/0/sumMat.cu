#include "includes.h"
__global__ void sumMat(double *A, double *B, double *C, int N)
{
int col = blockDim.x*blockIdx.x + threadIdx.x;
int row = blockDim.y*blockIdx.y + threadIdx.y;

if( (col < N) && (row < N)){
C[col*N + row] = A[col*N + row] + B[col*N + row];
//C[col][row] = B[col][row] + A[col][row];
}

}