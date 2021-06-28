#include "includes.h"
__global__ void outerProductSmartBruteForce(float* resultMatrix, float* vec, int vectorLength)
{
int col = (blockIdx.x * blockDim.x) + threadIdx.x; //column
int row = (blockIdx.y * blockDim.y) + threadIdx.y; //row


//check bounds
if(row >= vectorLength || col >= vectorLength || row > col)
return;

int index = (row * vectorLength + col) - (row * (row + 1)) / 2;

resultMatrix[index] += vec[row] * vec[col];

}