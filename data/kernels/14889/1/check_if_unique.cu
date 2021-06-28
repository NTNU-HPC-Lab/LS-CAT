#include "includes.h"
__global__ void check_if_unique(const unsigned *keys, unsigned       *is_unique, size_t          kSize) {
unsigned id = threadIdx.x +
blockIdx.x * blockDim.x +
blockIdx.y * blockDim.x * gridDim.x;
if (id == 0) {
is_unique[0] = 1;
} else if (id < kSize) {
is_unique[id] = (keys[id] != keys[id - 1] ? 1 : 0);
}
}