#include "includes.h"
__global__ void getMask(unsigned int *d_in, unsigned int *d_out, unsigned int in_size, unsigned int bit_shift, unsigned int One) {
unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
unsigned int bit = 0;
if (index < in_size) {
bit = d_in[index] & (1 << bit_shift);
bit = (bit > 0) ? 1 : 0;
d_out[index] = (One ? bit : 1 - bit);
}
}