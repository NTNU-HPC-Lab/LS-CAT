#include "includes.h"
__global__ void matrixAddKernel3(float* ans, float* M, float* N, int size) {
int col = blockIdx.x*blockDim.x + threadIdx.x;
if(col < size) {
for(int i = 0; i < size; ++i)
ans[i*size + col] = M[i*size + col] + N[i*size + col];
}
}