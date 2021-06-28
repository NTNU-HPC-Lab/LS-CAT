#include "includes.h"
__global__ void setDiagonalToZero(float* d_matrix, uint64_t columnsAndRows)
{
int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

//Check for out of bounds
if(absoluteThreadIdx >= columnsAndRows)
return;

//set diagonal element to zero
int matrixIndex = absoluteThreadIdx * columnsAndRows + absoluteThreadIdx;
d_matrix[matrixIndex] = 0;
}