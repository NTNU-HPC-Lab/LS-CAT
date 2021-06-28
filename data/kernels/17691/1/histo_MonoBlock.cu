#include "includes.h"
__global__ void histo_MonoBlock( unsigned char *buffer,long size,unsigned int *histo ) {

__shared__ unsigned int temp[256];
temp[threadIdx.x] = 0;
__syncthreads();

int i = threadIdx.x , offset = blockDim.x;
while (i < size) {
atomicAdd( &temp[buffer[i]], 1);
i += offset;
}
__syncthreads();

atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );

}