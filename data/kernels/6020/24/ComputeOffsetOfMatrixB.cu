#include "includes.h"
__global__ void ComputeOffsetOfMatrixB(const int32_t* row_sum, int32_t* output, int32_t N) {
for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
*(output + blockIdx.x * N + i) = -row_sum[blockIdx.x];
}
}