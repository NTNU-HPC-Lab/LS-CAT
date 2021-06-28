#include "includes.h"
__global__ void awkward_Content_getitem_next_missing_jagged_getmaskstartstop_kernel( int64_t* prefixed_index, int64_t* index_in, int64_t* offsets_in, int64_t* mask_out, int64_t* starts_out, int64_t* stops_out, int64_t length) {
int64_t block_id =
blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
int64_t thread_id = block_id * blockDim.x + threadIdx.x;

if(thread_id < length) {
int64_t pre_in = prefixed_index[thread_id] - 1;
starts_out[thread_id] = offsets_in[pre_in];

if (index_in[thread_id] < 0) {
mask_out[thread_id] = -1;
stops_out[thread_id] = offsets_in[pre_in];
} else {
mask_out[thread_id] = thread_id;
stops_out[thread_id] = offsets_in[pre_in + 1];
}
}
}