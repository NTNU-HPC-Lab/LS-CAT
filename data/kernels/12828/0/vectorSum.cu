#include "includes.h"
//Alfred Shaker
//10-13-2015



// CUDA kernel

__global__ void vectorSum(int *a, int *b, int *c, int n)
{
//get the id of global thread
int id = blockIdx.x*blockDim.x+threadIdx.x;

//checks to make sure we're not out of bounds
if(id < n)
c[id] = a[id] + b[id];

}