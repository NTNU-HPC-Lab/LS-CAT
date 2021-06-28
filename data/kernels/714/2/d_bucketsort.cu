#include "includes.h"
__global__ void d_bucketsort(unsigned int * d_in, unsigned int * d_indices, unsigned int * d_sublist, unsigned int * r_outputlist, unsigned int * d_bucketoffsets, int itemCount) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < itemCount) {
int newpos = d_bucketoffsets[d_sublist[idx]] + d_indices[idx];
r_outputlist[newpos] = d_in[idx];
}
}