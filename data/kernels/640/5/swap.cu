#include "includes.h"




__global__ void swap(unsigned int *in, unsigned int *in_pos, unsigned int *out, unsigned int *out_pos, unsigned int n)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < n)
{
in[i] = in[i] ^ out[i];
out[i] = in[i] ^ out[i];
in[i] = in[i] ^ out[i];

in_pos[i] = in_pos[i] ^ out_pos[i];
out_pos[i] = in_pos[i] ^ out_pos[i];
in_pos[i] = in_pos[i] ^ out_pos[i];
}
}