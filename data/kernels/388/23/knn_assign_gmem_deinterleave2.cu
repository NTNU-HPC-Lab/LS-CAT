#include "includes.h"
__global__ void knn_assign_gmem_deinterleave2( uint32_t length, uint16_t k, uint32_t *neighbors) {
volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
sample *= 2;
if (sample >= length) {
return;
}
for (int i = 0; i < k; i++) {
neighbors[sample * k + i] = neighbors[(length + sample) * k + k + i];
}
}