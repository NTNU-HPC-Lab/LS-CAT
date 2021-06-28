#include "includes.h"
__global__ void add_kernel(double *vals, double *ans, int N, double mu)
{
// Taken from geco.mines.edu/workshop/aug2010/slides/fri/cuda1.pd
int myblock = blockIdx.x + blockIdx.y * gridDim.x;
/* how big is each block within a grid */
int blocksize = blockDim.x * blockDim.y * blockDim.z;
/* get thread within a block */
int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;

int idx = myblock * blocksize + subthread;

if(idx < N)
ans[idx] = vals[idx] + mu;
}