#include "includes.h"
__global__ void matrixAddKernel1(float* ans, float* M, float* N, int size) {
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;
if((row < size) && (col < size)) {
ans[row*size + col] = M[row*size + col] + N[row*size + col];
}
}