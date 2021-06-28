#include "includes.h"
__global__ void pick_minus_log_ps(float *matrix, float *minus_log_ps, unsigned int *indices, unsigned int row, unsigned int col) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < row)
minus_log_ps[index] = -log(matrix[index * col + indices[index]]);
}