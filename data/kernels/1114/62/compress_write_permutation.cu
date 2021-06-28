#include "includes.h"
__global__ void compress_write_permutation(int *d_write_permutation, int *d_full_write_permutation, int *d_gcs, int total_pad_row_num, int chunk)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= total_pad_row_num) {
return;
}

int chunk_id = i / chunk;
if (d_gcs[chunk_id + 1] - d_gcs[chunk_id] > 0) {
int tid = i % chunk;
d_write_permutation[d_gcs[chunk_id] * chunk + tid] = d_full_write_permutation[i];
}
}