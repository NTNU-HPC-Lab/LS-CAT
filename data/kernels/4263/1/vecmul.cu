#include "includes.h"
__global__ void vecmul(float *A, float* B, float *C, int size)
{
// Row and Column indexes:
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;

// Are they bellow the maximum?
if (col < size && row < size) {
float result = 0;
for(int ix=0;ix<size;ix++) {
result += A[row*size+ix]*B[ix*size+col];
}
C[row*size+col] = result;
}
}