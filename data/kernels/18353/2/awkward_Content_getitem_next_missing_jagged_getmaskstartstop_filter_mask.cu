#include "includes.h"
__global__ void awkward_Content_getitem_next_missing_jagged_getmaskstartstop_filter_mask( int64_t* index_in, int64_t* filtered_index, int64_t length) {
int64_t block_id =
blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
int64_t thread_id = block_id * blockDim.x + threadIdx.x;

if(thread_id < length) {
if (index_in[thread_id] >= 0) {
filtered_index[thread_id] = 1;
}
}
}