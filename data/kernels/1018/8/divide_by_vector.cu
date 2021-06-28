#include "includes.h"
__global__ void divide_by_vector(float *matrix, float *vector, unsigned int row, unsigned int col) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < row * col)
matrix[index] /= vector[index / col];
}