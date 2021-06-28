#include "includes.h"
__global__ void scatter(unsigned int *d_in, unsigned int *d_index, unsigned int *d_out, unsigned int in_size) {
unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
if (index < in_size) {
d_out[d_index[index]] = d_in[index];
}
}