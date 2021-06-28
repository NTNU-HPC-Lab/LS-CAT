#include "includes.h"
__global__ void histKernel(char *inData, long size, unsigned int *histo)
{
__shared__ unsigned int temp[BIN_COUNT][BIN_COUNT];
__shared__ unsigned int blockSum[BIN_COUNT];
int i = 0;

while(i < BIN_COUNT)
temp[i++][threadIdx.x] = 0;

__syncthreads();

int tid = threadIdx.x + blockIdx.x * blockDim.x;
int offset = blockDim.x * gridDim.x;

while(tid < size) {
temp[(int)inData[tid]][threadIdx.x]++;
tid += offset;
}

__syncthreads();

i = 0;
while(i < BIN_COUNT)
blockSum[threadIdx.x] += temp[threadIdx.x][i++];

__syncthreads();

atomicAdd(&(histo[threadIdx.x]), blockSum[threadIdx.x]);
}