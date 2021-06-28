#include "includes.h"
__global__ void minus_one(float *matrix, unsigned int *indices, unsigned int row, unsigned int col) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < row)
matrix[index * col + indices[index]] -= 1;
}