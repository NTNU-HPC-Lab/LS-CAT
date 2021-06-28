#include "includes.h"




__global__ void exclusive_scan(unsigned int *in,unsigned int *out, int n)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < n)
{
out[i] -= in[i];
}
}