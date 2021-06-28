#include "includes.h"
__global__ void value_count_kernel_1(int nnz, const size_t *hash_value_index_sort, uint32_t *new_hash_value_flag) {
for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < nnz; gid += blockDim.x * gridDim.x) {
size_t cur_value = hash_value_index_sort[gid];
if (gid > 0) {
size_t former_value = hash_value_index_sort[gid - 1];
// decide if this is the start of a group(the elements in this group have the same
// hash_value_index_sort)
if (cur_value != former_value) {
new_hash_value_flag[gid] = 1;
} else {
new_hash_value_flag[gid] = 0;
}
} else {  // gid == 0
new_hash_value_flag[gid] = 1;
}
}
}