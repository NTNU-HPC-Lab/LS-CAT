#include "includes.h"
__global__ void matrixMulKernel(float* ans, float* M, float* N, int size) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if(row < size && col < size) {
float pVal = 0;
for (int i = 0; i < size; ++i)
pVal += M[row*size + i] * N[i*size + col];
ans[row*size + col] = pVal;
}
}