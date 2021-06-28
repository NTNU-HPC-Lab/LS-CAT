#include "includes.h"
__global__ void kernel_init_column_matrix(int *matrix, size_t pitch)
{
uint xOffset = (blockIdx.x * blockDim.x) + threadIdx.x;
uint yOffset = (blockIdx.y * blockDim.y) + threadIdx.y;

uint skipX = gridDim.x * blockDim.x;
uint skipY = gridDim.y * blockDim.y;

while (xOffset < colCount)
{
while (yOffset < rowCount)
{
int *memoryRow = (int *)((char *)matrix + (xOffset * pitch));
memoryRow[yOffset] = (xOffset * rowCount) + yOffset;

yOffset += skipY;
}
xOffset += skipX;
}
}