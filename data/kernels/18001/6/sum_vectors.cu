#include "includes.h"
__global__ void sum_vectors(uint32_t * src, uint32_t * dst, size_t N) {
size_t pos = threadIdx.x + blockDim.x * blockIdx.x;
if (pos < N) {
if (src[pos]) {
atomicAdd(dst + pos, src[pos]);
}
}
}