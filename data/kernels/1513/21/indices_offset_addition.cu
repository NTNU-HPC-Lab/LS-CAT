#include "includes.h"
__global__ void indices_offset_addition(int64_t *indices, int64_t *offsets, int64_t *output_indices, int batch_size) {
const int fea_count = 26;
__shared__ int64_t smem_offsets[fea_count];

if (threadIdx.x < fea_count) {
smem_offsets[threadIdx.x] = offsets[threadIdx.x];
}
__syncthreads();

int start_idx = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = start_idx; i < (batch_size * fea_count); i+=(gridDim.x * blockDim.x)) {
output_indices[i] = indices[i] + smem_offsets[i % fea_count];
}
}