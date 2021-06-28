#include "includes.h"

__global__ void cumo_na_index_aref_naview_index_index_beg_step_kernel(size_t *idx, size_t *idx1, size_t beg, ssize_t step, uint64_t n)
{
for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
idx[i] = idx1[beg + step * i];
}
}