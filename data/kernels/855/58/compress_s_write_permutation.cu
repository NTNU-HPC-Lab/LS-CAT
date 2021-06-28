#include "includes.h"
__global__ void compress_s_write_permutation(unsigned short *d_s_write_permutation, unsigned short *d_s_write_permutation_offset, int *d_write_permutation, int c_size, int chunk)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= c_size * chunk) {
return;
}

int chunk_id = i / chunk;
d_s_write_permutation[i] = (unsigned short)(d_write_permutation[i] % USHORT_MAX);
if (i % chunk == 0) {
d_s_write_permutation_offset[chunk_id] = (unsigned short)(d_write_permutation[i] / USHORT_MAX);
}
}