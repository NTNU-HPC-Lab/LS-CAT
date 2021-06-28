#include "includes.h"
__global__ void knn_assign_gmem_deinterleave1( uint32_t length, uint16_t k, uint32_t *neighbors) {
volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
if (sample >= length) {
return;
}
if (sample % 2 == 1) {
for (int i = 0; i < k; i++) {
neighbors[sample * k + i] = neighbors[sample * 2 * k + i];
}
} else {
for (int i = 0; i < k; i++) {
neighbors[(length + sample) * k + k + i] = neighbors[sample * 2 * k + i];
}
}
}