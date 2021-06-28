#include "includes.h"
__global__ void CudaGetBitAndOfRows(unsigned int* table1D, unsigned int* row, int rowSize, int tableRowCount)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx < tableRowCount * rowSize)
{
table1D[idx] = table1D[idx] & row[idx % rowSize];
}
}