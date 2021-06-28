#include "includes.h"
__global__ void DiamondStep(uint8_t* matrix, unsigned *random, int currentSize, int matrixSize, int randValue)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int half = currentSize / 2;
int minRand = -randValue;
int row = y * currentSize + half;
int col = x * currentSize + half;
int value;
value = (matrix[(row - half)*matrixSize + (col - half)] +
matrix[(row - half)*matrixSize + (col + half)] +
matrix[(row + half)*matrixSize + (col - half)] +
matrix[(row + half)*matrixSize + (col + half)] +
//VERSION 1
//(random[x*gridDim.x+y] % (randValue - minRand) + minRand)) / 4;
//VERSION 2
(random[row*matrixSize + col] % (randValue - minRand) + minRand)) / 4;
matrix[row*matrixSize + col] = value;
}