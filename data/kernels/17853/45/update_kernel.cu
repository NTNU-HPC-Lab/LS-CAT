#include "includes.h"
__global__ void update_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size, const size_t *deltaw_hash_value_index, const float *deltaw, float *hash_table_value) {
int tid = threadIdx.x;
int bid = blockIdx.x;

if ((bid < hash_value_index_count_num) && (tid < embedding_vec_size)) {
size_t value_index = deltaw_hash_value_index[bid];
size_t feature_index = value_index * embedding_vec_size + tid;
hash_table_value[feature_index] += deltaw[bid * embedding_vec_size + tid];
}
}