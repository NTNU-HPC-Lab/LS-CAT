#include "includes.h"
__global__ void PadMatrixInLeadingDimensionKernel(const int8_t* src, int8_t* dst, int col_src, int col_dst) {
for (int32_t i = threadIdx.x; i < col_src; i += blockDim.x) {
*(dst + blockIdx.x * col_dst + i) = *(src + blockIdx.x * col_src + i);
}
}