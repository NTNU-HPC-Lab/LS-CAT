#include "includes.h"
#define SIZE 2048*1024
#define BLOCKS 1000
#define THREADS 256

__global__ void histo_MultiBlock( unsigned char *buffer,long size,unsigned int *histo ) {

__shared__ unsigned int temp[256];
int i = threadIdx.x + blockIdx.x * THREADS;
int offset= THREADS * BLOCKS;
int memoffset = blockIdx.x * THREADS;
if(threadIdx.x <256)
temp[threadIdx.x] = 0;
__syncthreads();

while(i<size){
atomicAdd( &temp[buffer[i]], 1);
i+=offset;
}
__syncthreads();
if(threadIdx.x <256)
atomicAdd( &(histo[threadIdx.x+memoffset]), temp[threadIdx.x] );
}