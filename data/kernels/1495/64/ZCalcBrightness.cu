#include "includes.h"
__global__ static void ZCalcBrightness(float* DataArray, float* BrightArray, int size, int rows, int cols, int startIndex)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id >= size * rows)						// 超出範圍
return;

// 算 Index
int sizeIndex = id / rows;
int rowIndex = id % rows;

BrightArray[id] = 0;
for (int i = startIndex; i < cols; i++)
{
int currentID = sizeIndex * rows * cols + rowIndex * cols + i;
BrightArray[id] += DataArray[currentID];
}
}