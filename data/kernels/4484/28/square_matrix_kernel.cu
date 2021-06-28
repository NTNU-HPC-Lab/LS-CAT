#include "includes.h"
__global__ void square_matrix_kernel(int32_t num_rows, int32_t num_cols, const float* feats, int32_t ldf, float* feats_sq, int32_t lds) {
for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < num_rows;
i += blockDim.y * gridDim.y) {
for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_cols;
j += blockDim.x * gridDim.x) {
float f = feats[i * ldf + j];
feats_sq[i * lds + j] = f * f;
}
}
}