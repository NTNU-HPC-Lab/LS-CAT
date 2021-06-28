#include "includes.h"
__global__ void matrixMultiply2(float* A, float* C, int size)
{
float sum = 0;
int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;

if(Col < size && Row < size) {
for (int k = 0; k < size; k++)
sum += A[k * size + Row] * A[k * size + Col];

C[Row * size + Col] = sum;
}
}