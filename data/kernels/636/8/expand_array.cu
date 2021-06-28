#include "includes.h"
__global__ void expand_array( unsigned char *d_in, unsigned char *d_out)
{
uint32_t offset = blockDim.x * blockIdx.x + threadIdx.x;
unsigned char *input = d_in+offset*5*sizeof(unsigned char);
unsigned char *output = d_out+offset*6*sizeof(unsigned char);

output[0] = input[0] >> 4;
output[1] = input[0] << 4 | input[1] >> 4;
output[2] = input[1] << 4 | input[2] >> 4;
output[3] = input[2] & 0xf;
output[4] = input[3];
output[5] = input[4];
}