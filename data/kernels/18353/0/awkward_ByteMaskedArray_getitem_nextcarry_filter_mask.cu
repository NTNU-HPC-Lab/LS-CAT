#include "includes.h"
__global__ void awkward_ByteMaskedArray_getitem_nextcarry_filter_mask(int8_t* mask, bool validwhen, int64_t length) {
int64_t block_id =
blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
int64_t thread_id = block_id * blockDim.x + threadIdx.x;

if(thread_id < length) {
if ((mask[thread_id] != 0) == validwhen) {
mask[thread_id] = 1;
}
}
}