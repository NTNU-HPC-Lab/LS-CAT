#include "includes.h"
__global__ void outerProductSmartBruteForceLessThreads(float* resultMatrix, float* vec, uint64_t vectorLength)
{
int col = (blockIdx.x * blockDim.x) + threadIdx.x; //column
int row = (blockIdx.y * blockDim.y) + threadIdx.y; //row

//check bounds
if(row >= vectorLength || col >= vectorLength)
return;

//transpose
if(row > col)
{
row = vectorLength - row;
col = row + col;
}

int index = (row * vectorLength + col) - (row * (row + 1)) / 2;

resultMatrix[index] = vec[row] * vec[col];
}