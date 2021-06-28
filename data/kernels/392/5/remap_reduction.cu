#include "includes.h"
// Copyright Douglas Goddard 2016
// Licensed under the MIT license


// shout out to salix alba, you're a wizard mate
// http://stackoverflow.com/a/39862297/1176872
__global__ void remap_reduction( uint32_t *d_reduction, uint32_t *d_mapping, uint32_t *old_d_ij_buf, uint32_t sum_prev_size, uint32_t prev_size, uint32_t *new_d_ij_buf, uint32_t new_size)
{
uint32_t t_index = blockDim.x * blockIdx.x + threadIdx.x;
if(t_index < prev_size) {
if(d_reduction[t_index]) {
uint32_t index = d_mapping[t_index];
uint32_t i = *(old_d_ij_buf+2*sum_prev_size+t_index);
uint32_t j = *(old_d_ij_buf+2*sum_prev_size+prev_size+t_index);

// sort pairs in first round
if(!sum_prev_size && j < i) {
i ^= j;
j ^= i;
i ^= j;
}

*(new_d_ij_buf+2*sum_prev_size+index) = i;
*(new_d_ij_buf+2*sum_prev_size+new_size+index) = j;
}
}
}