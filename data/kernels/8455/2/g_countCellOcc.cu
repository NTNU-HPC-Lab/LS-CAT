#include "includes.h"
__global__ void g_countCellOcc(uint *_hash, uint *_cellOcc, uint _pixCount, uint _hashCellCount)
{
uint idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < _pixCount && _hash[idx] < _hashCellCount)
atomicAdd(&(_cellOcc[_hash[idx]]), 1);
}