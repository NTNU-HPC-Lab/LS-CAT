#include "includes.h"
__global__ void diffKernel( float *in, float *out, int n )
{
// Wrtie the kernel to implement the diff operation on an array
int id = (blockDim.x * blockIdx.x) + threadIdx.x;
if(id < n-1)
out[id] = in[id+1] - in[id];

}