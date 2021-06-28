#include "includes.h"
__global__ void squareFunc(unsigned int *d_in, unsigned int *d_out)
{
int idx = threadIdx.x;
unsigned int val = d_in[idx];
d_out[idx] = val * val;
//printf("%d square value %d \n  ", idx, d_out[idx]);
}