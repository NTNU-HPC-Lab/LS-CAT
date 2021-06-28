#include "includes.h"

__global__ void cumo_na_index_aref_naview_index_stride_kernel(size_t *idx, ssize_t s1, uint64_t n)
{
for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
idx[i] = idx[i] * s1;
}
}