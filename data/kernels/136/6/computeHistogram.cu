#include "includes.h"
__global__ void computeHistogram(unsigned int  *buffer, int size, unsigned int *histo )
{
__shared__ unsigned int temp[1024];

temp[threadIdx.x + 0] = 0;
temp[threadIdx.x + 256] = 0;
temp[threadIdx.x + 512] = 0;
temp[threadIdx.x + 768] = 0;
__syncthreads();

int i = threadIdx.x + blockIdx.x * blockDim.x;
int offset = blockDim.x * gridDim.x;
while (i < size)
{
atomicAdd( &temp[buffer[i]], 1);
i += offset;
}
__syncthreads();


atomicAdd( &(histo[threadIdx.x + 0]), temp[threadIdx.x + 0] );
atomicAdd( &(histo[threadIdx.x + 256]), temp[threadIdx.x + 256] );
atomicAdd( &(histo[threadIdx.x + 512]), temp[threadIdx.x + 512] );
atomicAdd( &(histo[threadIdx.x + 768]), temp[threadIdx.x + 768] );

}