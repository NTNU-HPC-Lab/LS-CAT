#include "includes.h"
__global__ void myhistKernel(unsigned char * buffer,unsigned int * histo)
{
__shared__ unsigned int temp[256];

int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int offset = x + y * blockDim.x * gridDim.x;

temp[threadIdx.x]=0;
__syncthreads();

atomicAdd( &temp[buffer[offset]], 1 );

__syncthreads();
atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}