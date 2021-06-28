#include "includes.h"
__global__ void InitializeMatrix_kernel( int8_t *matrix, int ldm, int rows, int columns) {

int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;

if (i < rows && j < columns) {
int offset = i + j * ldm;

matrix[offset] = 0;
if (i >= rows - 2 && j < 1) {
matrix[offset] = 0x0;
}
if (i < 1 && j >= columns - 2) {
matrix[offset] = 0x0;
}
}
}