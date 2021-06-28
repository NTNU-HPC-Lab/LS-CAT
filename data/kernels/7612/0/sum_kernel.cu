#include "includes.h"
__global__ void sum_kernel(float *g_odata, float *g_idata, int n)
{
// the size is determined by the host application
extern  __shared__  float sdata[];

// access thread id
const unsigned int tid = threadIdx.x;
// access number of threads in this block
//const unsigned int num_threads = blockDim.x;

// read in input data from global memory
sdata[2*tid] = g_idata[2*tid];
sdata[2*tid+1] = g_idata[2*tid+1];

//  printf ("KERNEL: sdata[%d] = %f\n", (2*tid), sdata[2*tid]);
//  printf ("KERNEL: sdata[%d] = %f\n", (2*tid), sdata[2*tid+1]);
__syncthreads();

// perform some computations
sdata[2*tid] = sdata[2*tid] + sdata[2*tid+1];
__syncthreads();

g_odata[tid]   = sdata[tid];

}