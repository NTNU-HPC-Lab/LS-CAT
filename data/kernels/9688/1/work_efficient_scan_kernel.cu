#include "includes.h"

#define THREADS 256
#define BLOCKS 32
#define NUM THREADS*BLOCKS

int seed_var =1239;

__global__ void work_efficient_scan_kernel(int *X, int *Y, int InputSize)
{
extern __shared__ int XY[];
int i= blockIdx.x*blockDim.x+ threadIdx.x;
if (i < InputSize)
{
XY[threadIdx.x] = X[i];
}
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
{
__syncthreads();
int index = (threadIdx.x+1) * 2* stride -1;
if (index < blockDim.x)
{
XY[index] += XY[index -stride];
}
}
for (int stride = THREADS/4; stride > 0; stride /= 2)
{
__syncthreads();
int index = (threadIdx.x+1)*stride*2 -1;
if(index + stride < THREADS)
{
XY[index + stride] += XY[index];
}
}
__syncthreads();
Y[i] = XY[threadIdx.x];

//OWN CODE
__syncthreads();
if(threadIdx.x < blockIdx.x)
{
XY[threadIdx.x] = Y[threadIdx.x*blockDim.x + (blockDim.x-1)];
}
__syncthreads();
for(unsigned int stride =0; stride < blockIdx.x; stride++)
{
Y[i] += XY[stride];
}
__syncthreads();
}