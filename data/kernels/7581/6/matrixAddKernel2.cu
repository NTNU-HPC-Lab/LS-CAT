#include "includes.h"
__global__ void matrixAddKernel2(float* ans, float* M, float* N, int size) {
int row = blockIdx.y*blockDim.y + threadIdx.y;
if(row < size) {
for(int i = 0; i < size; ++i)
ans[row*size + i] = M[row*size + i] + N[row*size + i];
}
}