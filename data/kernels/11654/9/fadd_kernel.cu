#include "includes.h"
__global__ void fadd_kernel(float *vals, int N, float mu)
{
// Taken from geco.mines.edu/workshop/aug2010/slides/fri/cuda1.pd
int myblock = blockIdx.x + blockIdx.y * gridDim.x;
/* how big is each block within a grid */
int blocksize = blockDim.x * blockDim.y * blockDim.z;
/* get thread within a block */
int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;

int idx = myblock * blocksize + subthread;

if(idx < N)
vals[idx] = mu;
}