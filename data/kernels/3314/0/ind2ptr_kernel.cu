#include "includes.h"
__global__ void ind2ptr_kernel(const int64_t *ind_data, int64_t *out_data, int64_t M, int64_t numel) {

int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

if (thread_idx == 0) {
for (int64_t i = 0; i <= ind_data[0]; i++)
out_data[i] = 0;
} else if (thread_idx < numel) {
for (int64_t i = ind_data[thread_idx - 1]; i < ind_data[thread_idx]; i++)
out_data[i + 1] = thread_idx;
} else if (thread_idx == numel) {
for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++)
out_data[i] = numel;
}
}