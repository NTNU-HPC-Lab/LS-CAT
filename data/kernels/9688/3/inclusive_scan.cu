#include "includes.h"
__global__ void inclusive_scan(const unsigned int *X, unsigned int *Y, int N)
{
extern __shared__ int XY[];
unsigned   int i = blockIdx.x * blockDim.x + threadIdx.x;
// load input into __shared__ memory
if(i<N)
{
XY[threadIdx.x] =X[i];
}
/*Note here stride <= threadIdx.x, means that everytime the threads with threadIdx.x less than
stride do not participate in loop*/
for(unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
__syncthreads();
XY[threadIdx.x]+= XY[threadIdx.x - stride];
}
/*This is executed by all threads, so that they store the final prefix sum to
corresponding locations in global   memory*/
Y[i]=XY[threadIdx.x];

// wait until all threads of this block writes the output for all prefix sum within the block
__syncthreads();
if (threadIdx.x < blockIdx.x) //for 1st block onwards
{
//update the shared memory to keep prefix sum of last elements of previous block's
XY[threadIdx.x] = Y[threadIdx.x * blockDim.x + BLOCK_SIZE - 1];
}
__syncthreads();
for (int stride = 0; stride < blockIdx.x; stride++)
{    //add all previous las elements to this block elements
Y[threadIdx.x + blockDim.x * blockIdx.x] += XY[stride];
__syncthreads();

}
}