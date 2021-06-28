#include "includes.h"
__global__ void Counting(int* HalfData, int HalfDataSize, int N)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i<HalfDataSize)
{
HalfData[i] *= N;
}
}