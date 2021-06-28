#include "includes.h"
__global__ void awkward_ByteMaskedArray_getitem_nextcarry_kernel(int64_t* prefixed_mask, int64_t* to_carry, int8_t* mask, int64_t length) {
int64_t block_id =
blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
int64_t thread_id = block_id * blockDim.x + threadIdx.x;

if(thread_id < length) {
if (mask[thread_id] != 0) {
to_carry[prefixed_mask[thread_id] - 1] = thread_id;
}
}
}