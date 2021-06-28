#include "includes.h"
__global__ void Mul(float *newMatrix,float *mulMatrix,int Max,float *sumMatrix){
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

// int Index = iy * nx + ix;

for (int k = 0; k < Max; k++) {
// Accumulate results for a single element
// c[row * nx + col] += a[row * nx + k] * b[k * nx + col];
// printf("C[%d] = a[%d] * b[%d]\n",row * nx + col,row * nx + k, k * nx + col);
atomicAdd(&mulMatrix[row * Max + col],newMatrix[row * Max + k] * newMatrix[k * Max + col]);
// atomicAdd(&sumMatrix[0],mulMatrix[row * Max + col]);
}
}