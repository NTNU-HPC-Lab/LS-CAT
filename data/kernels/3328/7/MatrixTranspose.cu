#include "includes.h"
__global__ void MatrixTranspose(const float *A_elements, float *B_elements, const int A_width, const int A_height)
{
int strideRow = blockDim.y * gridDim.y;
int strideCol = blockDim.x * gridDim.x;

for(int row = blockIdx.y * blockDim.y + threadIdx.y; row < A_width; row += strideRow)
for(int col = blockIdx.x * blockDim.x + threadIdx.x; col < A_height; col += strideCol)
{
B_elements[row * A_height + col] = A_elements[col * A_width + row];
}
}