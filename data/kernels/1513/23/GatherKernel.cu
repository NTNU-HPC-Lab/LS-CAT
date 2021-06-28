#include "includes.h"
__global__ void GatherKernel(const float* params, int64_t num_features, int embed_size, int batch_size, int query_nnz, const int64_t* indices, float* ret) {
int tid = threadIdx.x, bid = blockIdx.x;

extern __shared__ int shmem_indices[];

// each CTA load one row of indices in the mini batch into shared memory
for (int i = tid; i < query_nnz; i += blockDim.x) {
shmem_indices[i] = indices[query_nnz * bid + i];
}
__syncthreads();

#pragma unroll
for (int i = 0; i < query_nnz; ++i) {
// printf("%d, %d, %d\n", bid, i, shmem_indices[i]);
ret[(bid * query_nnz + i) * embed_size + tid] =
params[(int64_t)shmem_indices[i] * embed_size + tid];
}
}