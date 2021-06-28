#include "includes.h"
__global__ void SquareStep(uint8_t* matrix, unsigned* random, int currentSize, int matrixSize, int maxRowThread, int maxColThread, int randValue)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
/*x/y can be greater than maxRowThread/maxColThread because the number
of created threads might not be multiple of the number of threads in a block*/
if (x < maxRowThread && y < maxColThread)
{
int half = currentSize / 2;
int minRand = -randValue;
int value = 0;
int div = 0;
int cond;
int elemX = x * currentSize*(y % 2 == 0) +
y * half*(y % 2 != 0);
int elemY = (y*half + half)*(y % 2 == 0) +
x * currentSize*(y % 2 != 0);
// CUDA VERSION 2: it uses conditions as variables
// to avoid divergent branches
cond = elemX != 0;
value += matrix[(elemX - half * cond) *
matrixSize + elemY] * cond;
div += cond;
cond = elemX != matrixSize - 1;
value += matrix[(elemX + half * cond) *
matrixSize + elemY] * cond;
div += cond;
cond = elemY != 0;
value += matrix[elemX * matrixSize +
elemY - half * cond] * cond;
div += cond;
cond = elemY != matrixSize - 1;
value += matrix[elemX*matrixSize + elemY
+ half * cond] * cond;
div += cond;
/*
// CUDA VERSION 1: it uses divergent branches
if (elemX != 0)
{
value += matrix[(elemX - half)*matrixSize + elemY];
div++;
}
if (elemX != matrixSize-1)
{
value += matrix[(elemX + half)*matrixSize + elemY];
div++;
}
if (elemY != 0)
{
value += matrix[elemX*matrixSize + elemY - half];
div++;
}
if (elemY != matrixSize-1)
{
value += matrix[elemX*matrixSize + elemY + half];
div++;
}*/
//VERSION 1: random index is correct for the
//first version of random generation but not for the second one
//value += (minRand + random[x*gridDim.x+y] % (randValue - minRand));
//VERSION 2
value += (minRand + random[elemX*matrixSize+elemY] % (randValue - minRand));
matrix[elemX*matrixSize + elemY] = value / div;
}
}