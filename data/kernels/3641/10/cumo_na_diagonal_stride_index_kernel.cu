#include "includes.h"
__global__ void cumo_na_diagonal_stride_index_kernel(size_t *idx, ssize_t s0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
idx[i] = s0*(i+k0) + idx1[i+k1];
}
}