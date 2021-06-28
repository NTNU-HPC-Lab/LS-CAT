#include "includes.h"

#define HISTOGRAM_LENGTH 256












__global__ void histo_kernel(unsigned char * buffer, unsigned int * histo, long size)
{
//  compute histogram with a private version in each block
__shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

int bx = blockIdx.x;
int tx = threadIdx.x;

//  index of current pixel
int index = tx+bx*blockDim.x;

//  set initial values of histogram to zero
if (tx < HISTOGRAM_LENGTH) histo_private[tx] = 0;

__syncthreads();


int stride = blockDim.x*gridDim.x;

//iterate to add values
while (index < stride)
{
atomicAdd(&(histo_private[buffer[index]]), 1);
index += stride;
}

__syncthreads();

//copy private histogram to device histogram
if(tx<256)
{
atomicAdd(&(histo[tx]), histo_private[tx]);
}
}