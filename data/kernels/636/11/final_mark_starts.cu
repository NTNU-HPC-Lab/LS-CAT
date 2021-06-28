#include "includes.h"
__global__ void final_mark_starts( uint8_t *hashes, uint32_t *sort_indices, uint32_t *off_map, uint32_t r, uint32_t hash_count)
{
uint32_t t_index = blockDim.x * blockIdx.x + threadIdx.x;
if(t_index < hash_count) {
uint32_t t_prev_index = (t_index-1) % hash_count; // wrap around at index 0

uint32_t index = sort_indices[t_index];
uint32_t prev_index = sort_indices[t_prev_index];

unsigned char* hash = hashes+index*30*sizeof(unsigned char)+r*3;
unsigned char* prev_hash = hashes+prev_index*30*sizeof(unsigned char)+r*3;

uint64_t key = ((uint64_t)hash[0]) << 40 | ((uint64_t)hash[1]) << 32 | hash[2] << 24;
key |= hash[3] << 16 | hash[4] << 8 | hash[5];

uint64_t prev_key = ((uint64_t)prev_hash[0]) << 40 | ((uint64_t)prev_hash[1]) << 32 | prev_hash[2] << 24;
prev_key |= prev_hash[3] << 16 | prev_hash[4] << 8 | prev_hash[5];

if((key ^ prev_key) != 0) {
off_map[t_index] = 1;
}
}
}