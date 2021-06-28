#include "includes.h"
__global__ void matrixMult(int* m, int* n, int* p, int size)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int p_sum;

for (int i = 0;i < size;i++) {
p_sum += m[row * size + i] * n[col * size + i];
}
p[row * size + col] = p_sum;
}