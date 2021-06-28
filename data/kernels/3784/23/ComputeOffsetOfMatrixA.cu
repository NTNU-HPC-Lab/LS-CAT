#include "includes.h"
__global__ void ComputeOffsetOfMatrixA(const int32_t* col_sum, int32_t* output, int32_t N) {
for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
*(output + blockIdx.x * N + i) = -col_sum[i];
}
}