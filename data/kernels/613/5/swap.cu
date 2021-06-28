#include "includes.h"
//Udacity HW 4
//Radix Sorting





__global__ void swap(unsigned int *in, unsigned int *in_pos, unsigned int *out, unsigned int *out_pos, unsigned int n)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < n)
{
unsigned int temp = in[i];
in[i] = out[i];
out[i] = temp;

temp = in_pos[i];
in_pos[i] = out_pos[i];
out_pos[i] = temp;
}
}