#include "includes.h"
__global__ void get_sort_keys( unsigned char* d_hashes, uint32_t *d_sort_keys, uint32_t *d_sort_indices, uint32_t r, uint32_t num_keys)
{
uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
if(index < num_keys) {
d_sort_indices[index] = index;

unsigned char* input = d_hashes+index*30*sizeof(unsigned char)+3*r;
uint32_t sort_key = input[0] << 16 | input[1] << 8 | input[2];
d_sort_keys[index] = sort_key;
}
}