#include "includes.h"




//#define array_size 100000000
#define array_size 101

//987459712


cudaError_t addWithCuda(int *total);

__shared__ int temp[array_size];

__global__ void addKernel(int *tid_c, int *tid_total)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
tid_c[tid] = tid;
if (tid <= array_size)
{

temp[threadIdx.x] = tid;

if (threadIdx.x==0)
{
for(int i=0;i<=blockDim.x;i++)
{
//__syncthreads();
atomicAdd(tid_total, temp[i]);
//__syncthreads();
//printf("i = %d \n", *tid_total);
}
}

}

}