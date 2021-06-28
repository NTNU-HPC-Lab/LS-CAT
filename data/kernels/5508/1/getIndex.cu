#include "includes.h"
__global__ void getIndex(unsigned int *d_index, unsigned int *d_scan, unsigned int *d_mask, unsigned int in_size, unsigned int total_pre) {
unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

if (index < in_size) {
if (d_mask[index] == 1) {
d_index[index] = total_pre + d_scan[index];
}
}
}