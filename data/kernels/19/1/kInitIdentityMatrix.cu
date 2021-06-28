#include "includes.h"
__global__ void kInitIdentityMatrix(float* a, int size, int num_elements) {
const int idxX = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
for (int x = idxX; x < num_elements; x += gridDim.x * THREADS_PER_BLOCK) {
if (x % size == x / size) {
a[x] = 1;
} else {
a[x] = 0;
}
}
}