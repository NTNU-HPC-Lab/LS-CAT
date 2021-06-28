#include "includes.h"
__global__ void prepare_indices(const unsigned num_keys, unsigned *data) {
unsigned index = threadIdx.x +
blockIdx.x * blockDim.x +
blockIdx.y * blockDim.x * gridDim.x;
if (index < num_keys) {
data[index] = index;
}
}