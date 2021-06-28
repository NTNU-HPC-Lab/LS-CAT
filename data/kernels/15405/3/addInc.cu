#include "includes.h"
__global__ void addInc(unsigned int* deviceInput, unsigned int* deviceOutput, int eleCnt, unsigned int* deviceInc)
{
/*
__shared__ int inc;
if (threadIdx.x == 0)
{
inc = deviceInc[blockIdx.x];
}
__syncthreads();
*/
int inc = deviceInc[blockIdx.x];

int cntInB = blockDim.x * 2;
int idxInG = blockIdx.x * cntInB + threadIdx.x;

if (idxInG < eleCnt)
{
deviceOutput[idxInG] = deviceInput[idxInG] + inc;
}

if (idxInG + blockDim.x < eleCnt)
{
deviceOutput[idxInG + blockDim.x] = deviceInput[idxInG + blockDim.x] + inc;
}

}