#include "includes.h"
// Copyright Douglas Goddard 2016
// Licensed under the MIT license


// shout out to salix alba, you're a wizard mate
// http://stackoverflow.com/a/39862297/1176872
__global__ void map_index_to_prefix( uint8_t *hashes, uint32_t *sort_indices, uint32_t *comb_count, uint32_t *comb_sum, uint32_t *comb_prefix, uint32_t r, uint32_t size)
{
uint32_t t_index = blockDim.x * blockIdx.x + threadIdx.x;
if(t_index < size) {
uint32_t index = sort_indices[t_index];
unsigned char* hash = hashes+index*30*sizeof(unsigned char)+r*3;
uint32_t key = hash[0] << 16 | hash[1] << 8 | hash[2];

uint32_t count = comb_count[key];
uint32_t sum = comb_sum[key];
for(int i=(sum-count); i<sum; i++) {
comb_prefix[i] = key;
}
}
}