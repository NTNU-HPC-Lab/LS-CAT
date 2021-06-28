#include "includes.h"
__global__ void ptr2ind_kernel(const int64_t *ptr_data, int64_t *out_data, int64_t E, int64_t numel) {

int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

if (thread_idx < numel) {
int64_t idx = ptr_data[thread_idx], next_idx = ptr_data[thread_idx + 1];
for (int64_t i = idx; i < next_idx; i++) {
out_data[i] = thread_idx;
}
}
}