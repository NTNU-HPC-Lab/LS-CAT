#include "includes.h"
__global__ void GatherBackwardFuseSgdKernel(const float* grads, int64_t num_features, int embed_size, int batch_size, int query_nnz, const int64_t* indices, float lr, float* params) {
int tid = threadIdx.x, bid = blockIdx.x;

extern __shared__ int shmem_indices[];

for (int i = tid; i < query_nnz; i += blockDim.x) {
shmem_indices[i] = indices[query_nnz * bid + i];
}
__syncthreads();

#pragma unroll
for (int i = 0; i < query_nnz; ++i) {
atomicAdd(&params[(int64_t)shmem_indices[i] * embed_size + tid],
-lr * grads[(bid * query_nnz + i) * embed_size + tid]);
}
}