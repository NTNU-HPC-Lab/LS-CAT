#include "includes.h"


__global__ void histo_kernal(char *buffer, long size, int *histo )
{
__shared__ int temp[256];
temp[threadIdx.x] = 0;
__syncthreads();

int i = threadIdx.x + blockIdx.x * blockDim.x;
int offset = blockDim.x * gridDim.x;
int z;
while (i < size)
{
z = buffer[i];
atomicAdd( &temp[z], 1);
i += offset;
}
__syncthreads();


atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}