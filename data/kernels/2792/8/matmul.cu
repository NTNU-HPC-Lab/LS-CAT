#include "includes.h"
__global__ void matmul(const float_t *A, const float_t *B, float_t *C, const float_t alpha, const float_t beta, int n, int d, int k, int max_block_rows) {
extern __shared__ __align__(sizeof(float_t)) unsigned char my_smem[];
float_t *shared = reinterpret_cast<float_t *>(my_smem);

float_t *s_A = shared;
float_t *s_B = shared + max_block_rows * d;

for (int i = threadIdx.x; i < d * k; i += blockDim.x) {
s_B[i] = B[i];
}

size_t block_start_row_index = blockIdx.x * max_block_rows;
size_t block_rows = max_block_rows;

if (blockIdx.x == gridDim.x - 1 && n % max_block_rows != 0) {
block_rows = n % max_block_rows;
}

for (size_t i = threadIdx.x; i < d * block_rows; i += blockDim.x) {
s_A[i] = alpha * A[d * block_start_row_index + i];
}

__syncthreads();

float_t elem_c = 0;

int col_c = threadIdx.x % k;
size_t abs_row_c = block_start_row_index + threadIdx.x / k;
int row_c = threadIdx.x / k;

// Thread/Block combination either too far for data array
// Or is calculating for index that should be calculated in a different
// blocks - in some edge cases "col_c * n + abs_row_c" can yield same
// result in different thread/block combinations
if (abs_row_c >= n || threadIdx.x >= block_rows * k) {
return;
}

for (size_t i = 0; i < d; i++) {
elem_c += s_B[d * col_c + i] * s_A[d * row_c + i];
}

C[col_c * n + abs_row_c] = beta * C[col_c * n + abs_row_c] + elem_c;
}